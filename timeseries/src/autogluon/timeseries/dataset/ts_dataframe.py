from __future__ import annotations
import os
import copy
import itertools
import logging
import reprlib
from collections.abc import Iterable
from itertools import islice
from scipy.signal import argrelextrema
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Type, overload
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from joblib.parallel import Parallel, delayed
from pandas.core.internals import ArrayManager, BlockManager  # type: ignore
from typing_extensions import Self
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stl._stl import STL
from scipy import stats
from autogluon.common.loaders import load_pd

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

plt.rcParams["font.sans-serif"] = ["AR PL UKai CN", "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号乱码问题

class TimeSeriesDataFrame(pd.DataFrame):
    """A collection of univariate time series, where each row is identified by an (``item_id``, ``timestamp``) pair.

    For example, a time series dataframe could represent the daily sales of a collection of products, where each
    ``item_id`` corresponds to a product and ``timestamp`` corresponds to the day of the record.

    Parameters
    ----------
    data : pd.DataFrame, str, pathlib.Path or Iterable
        Time series data to construct a ``TimeSeriesDataFrame``. The class currently supports four input formats.

        1. Time series data in a pandas DataFrame format without multi-index. For example::

                   item_id  timestamp  target
                0        0 2019-01-01       0
                1        0 2019-01-02       1
                2        0 2019-01-03       2
                3        1 2019-01-01       3
                4        1 2019-01-02       4
                5        1 2019-01-03       5
                6        2 2019-01-01       6
                7        2 2019-01-02       7
                8        2 2019-01-03       8

        You can also use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.from_data_frame` for loading data in such format.

        2. Path to a data file in CSV or Parquet format. The file must contain columns ``item_id`` and ``timestamp``, as well as columns with time series values. This is similar to Option 1 above (pandas DataFrame format without multi-index). Both remote (e.g., S3) and local paths are accepted. You can also use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.from_path` for loading data in such format.

        3. Time series data in pandas DataFrame format with multi-index on ``item_id`` and ``timestamp``. For example::

                                    target
                item_id timestamp
                0       2019-01-01       0
                        2019-01-02       1
                        2019-01-03       2
                1       2019-01-01       3
                        2019-01-02       4
                        2019-01-03       5
                2       2019-01-01       6
                        2019-01-02       7
                        2019-01-03       8

        4. Time series data in Iterable format. For example::

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Period("01-01-2019", freq='D')}
                ]

        You can also use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.from_iterable_dataset` for loading data in such format.

    static_features : pd.DataFrame, str or pathlib.Path, optional
        An optional dataframe describing the metadata of each individual time series that does not change with time.
        Can take real-valued or categorical values. For example, if ``TimeSeriesDataFrame`` contains sales of various
        products, static features may refer to time-independent features like color or brand.

        The index of the ``static_features`` index must contain a single entry for each item present in the respective
        ``TimeSeriesDataFrame``. For example, the following ``TimeSeriesDataFrame``::

                                target
            item_id timestamp
            A       2019-01-01       0
                    2019-01-02       1
                    2019-01-03       2
            B       2019-01-01       3
                    2019-01-02       4
                    2019-01-03       5

        is compatible with the following ``static_features``::

                     feat_1 feat_2
            item_id
            A           2.0    bar
            B           5.0    foo

        ``TimeSeriesDataFrame`` will ensure consistency of static features during serialization/deserialization, copy
        and slice operations.

        If ``static_features`` are provided during ``fit``, the ``TimeSeriesPredictor`` expects the same metadata to be
        available during prediction time.
    id_column : str, optional
        Name of the ``item_id`` column, if it's different from the default. This argument is only used when
        constructing a TimeSeriesDataFrame using format 1 (DataFrame without multi-index) or 2 (path to a file).
    timestamp_column : str, optional
        Name of the ``timestamp`` column, if it's different from the default. This argument is only used when
        constructing a TimeSeriesDataFrame using format 1 (DataFrame without multi-index) or 2 (path to a file).
    num_cpus : int, default = -1
        Number of CPU cores used to process the iterable dataset in parallel. Set to -1 to use all cores. This argument
        is only used when constructing a TimeSeriesDataFrame using format 4 (iterable dataset).

    """

    index: pd.MultiIndex  # type: ignore
    _metadata = ["_static_features"]

    IRREGULAR_TIME_INDEX_FREQSTR: Final[str] = "IRREG"
    ITEMID: Final[str] = "item_id"
    TIMESTAMP: Final[str] = "timestamp"

    def __init__(
        self,
        data: pd.DataFrame | str | Path | Iterable,
        static_features: pd.DataFrame | str | Path | None = None,
        id_column: str | None = None,
        timestamp_column: str | None = None,
        num_cpus: int = -1,
        *args,
        **kwargs,
    ):
        if isinstance(data, (BlockManager, ArrayManager)):
            # necessary for copy constructor to work in pandas <= 2.0.x. In >= 2.1.x this is replaced by _constructor_from_mgr
            pass
        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                self._validate_multi_index_data_frame(data)
            else:
                data = self._construct_tsdf_from_data_frame(
                    data, id_column=id_column, timestamp_column=timestamp_column
                )
        elif isinstance(data, (str, Path)):
            data = self._construct_tsdf_from_data_frame(
                load_pd.load(str(data)), id_column=id_column, timestamp_column=timestamp_column
            )
        elif isinstance(data, Iterable):
            data = self._construct_tsdf_from_iterable_dataset(data, num_cpus=num_cpus)
        else:
            raise ValueError(f"data must be a pd.DataFrame, Iterable, string or Path (received {type(data)}).")
        super().__init__(data=data, *args, **kwargs)  # type: ignore
        self._static_features: pd.DataFrame | None = None
        if static_features is not None:
            self.static_features = self._construct_static_features(static_features, id_column=id_column)

    @property
    def _constructor(self) -> Type[TimeSeriesDataFrame]:
        return TimeSeriesDataFrame

    def _constructor_from_mgr(self, mgr, axes):
        # Use the default constructor when constructing from _mgr. Otherwise pandas enters an infinite recursion by
        # repeatedly calling TimeSeriesDataFrame constructor
        df = self._from_mgr(mgr, axes=axes)    # type: ignore
        df._static_features = self._static_features
        return df

    @classmethod
    def _construct_tsdf_from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: str | None = None,
        timestamp_column: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        if id_column is not None:
            assert id_column in df.columns, f"Column '{id_column}' not found!"
            if id_column != cls.ITEMID and cls.ITEMID in df.columns:
                logger.warning(
                    f"Renaming existing column '{cls.ITEMID}' -> '__{cls.ITEMID}' to avoid name collisions."
                )
                df.rename(columns={cls.ITEMID: "__" + cls.ITEMID}, inplace=True)
            df.rename(columns={id_column: cls.ITEMID}, inplace=True)

        if timestamp_column is not None:
            assert timestamp_column in df.columns, f"Column '{timestamp_column}' not found!"
            if timestamp_column != cls.TIMESTAMP and cls.TIMESTAMP in df.columns:
                logger.warning(
                    f"Renaming existing column '{cls.TIMESTAMP}' -> '__{cls.TIMESTAMP}' to avoid name collisions."
                )
                df.rename(columns={cls.TIMESTAMP: "__" + cls.TIMESTAMP}, inplace=True)
            df.rename(columns={timestamp_column: cls.TIMESTAMP}, inplace=True)

        if cls.TIMESTAMP in df.columns:
            df[cls.TIMESTAMP] = pd.to_datetime(df[cls.TIMESTAMP])

        cls._validate_data_frame(df)
        return df.set_index([cls.ITEMID, cls.TIMESTAMP])

    @classmethod
    def _construct_tsdf_from_iterable_dataset(cls, iterable_dataset: Iterable, num_cpus: int = -1) -> pd.DataFrame:
        def load_single_item(item_id: int, ts: dict) -> pd.DataFrame:
            start_timestamp = ts["start"]
            freq = start_timestamp.freq
            if isinstance(start_timestamp, pd.Period):
                start_timestamp = start_timestamp.to_timestamp(how="S")
            target = ts["target"]
            datetime_index = tuple(pd.date_range(start_timestamp, periods=len(target), freq=freq))
            idx = pd.MultiIndex.from_product([(item_id,), datetime_index], names=[cls.ITEMID, cls.TIMESTAMP])
            return pd.Series(target, name="target", index=idx).to_frame()

        cls._validate_iterable(iterable_dataset)
        all_ts = Parallel(n_jobs=num_cpus)(
            delayed(load_single_item)(item_id, ts) for item_id, ts in enumerate(iterable_dataset)
        )
        return pd.concat(all_ts)

    @classmethod
    def _validate_multi_index_data_frame(cls, data: pd.DataFrame):
        """Validate a multi-index pd.DataFrame can be converted to TimeSeriesDataFrame"""

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(data)}")
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"data must have pd.MultiIndex, got {type(data.index)}")
        if not pd.api.types.is_datetime64_dtype(data.index.dtypes[cls.TIMESTAMP]):
            raise ValueError(f"for {cls.TIMESTAMP}, the only pandas dtype allowed is `datetime64`.")
        if not data.index.names == (f"{cls.ITEMID}", f"{cls.TIMESTAMP}"):
            raise ValueError(
                f"data must have index names as ('{cls.ITEMID}', '{cls.TIMESTAMP}'), got {data.index.names}"
            )
        item_id_index = data.index.levels[0]
        if not (pd.api.types.is_integer_dtype(item_id_index) or pd.api.types.is_string_dtype(item_id_index)):
            raise ValueError(f"all entries in index `{cls.ITEMID}` must be of integer or string dtype")


    def _compute_missing_rate_KANG(self) -> dict:
        """
        计算每个 item_id 在所有列上的缺失率，仅用于日志输出
        """

        try:
            results = []
            missing_dict = {}

            for col in self.columns:
                try:
                    missing_rate = (
                        self[col]
                        .isna()
                        .groupby(level=self.ITEMID)
                        .mean()
                    )

                    results.append(pd.DataFrame({
                        "item_id": missing_rate.index,
                        "column": col,
                        "missing_rate": missing_rate.values
                    }))

                    # 用于返回的字典
                    for item_id, rate in missing_rate.items():
                        missing_dict[(item_id, col)] = float(rate)

                except Exception as e:
                    logger.warning(
                        f"Failed to compute missing rate for column '{col}': {e}"
                    )

            if not results:
                logger.warning("No missing rate statistics were computed.")

            # 合并结果
            missing_df = pd.concat(results, ignore_index=True)

            # 取缺失率最高的 Top N
            top = (
                missing_df
                .sort_values("missing_rate", ascending=False)
            )

            logger.info("Top missing rates by item_id and column:")
            for i, row in enumerate(top.itertuples(index=False), start=1):
                logger.info(
                    f"  {i}. Item {row.item_id}, Column '{row.column}': "
                    f"{row.missing_rate:.2%}"
                )

            return missing_dict

        except Exception as e:
            logger.warning(f"Could not compute missing rate: {e}")
            return {}
        

    def _plot_distribution_KANG(
        self,
        bins: int = 50,
        figsize: tuple = (10, 6),
        save: bool = True,
        save_dir: str = "results/plot",
        dpi: int = 300
    ) -> dict:
        """
        为所有列绘制分布直方图（每列一张），并返回直方图数据

        Returns
        -------
        dict
            {
                column_name: {
                    "bins": np.ndarray,
                    "counts": np.ndarray,
                    "total_points": int
                }
            }
        """

        hist_dict = {}

        try:
            os.makedirs(save_dir, exist_ok=True)

            for col in self.columns:
                try:
                    data = self[col].dropna()
                    if data.empty:
                        continue

                    # ======================
                    # 0. 先用 numpy 计算直方图（这是“数据源”）
                    # ======================
                    counts, bin_edges = np.histogram(data, bins=bins)

                    hist_dict[col] = {
                        "bins": bin_edges,
                        "counts": counts,
                        "total_points": int(len(data))
                    }

                    # ======================
                    # 1. 开始画图
                    # ======================
                    fig, axes = plt.subplots(1, 2, figsize=figsize)

                    # ---- 总体分布（用同一套 bins）
                    axes[0].hist(
                        data,
                        bins=bin_edges,
                        edgecolor="black",
                        alpha=0.7
                    )
                    axes[0].set_xlabel(col)
                    axes[0].set_ylabel("Frequency")
                    axes[0].set_title(f"Overall Distribution of {col}")
                    axes[0].grid(True, alpha=0.3)

                    # ======================
                    # 2. 按 item_id 分布（抽样）
                    # ======================
                    if self.num_items > 10:
                        sampled_items = self.item_ids[:10]
                    else:
                        sampled_items = self.item_ids

                    for item_id in sampled_items:
                        try:
                            item_data = self.loc[item_id][col].dropna()
                            if not item_data.empty:
                                axes[1].hist(
                                    item_data,
                                    bins=bin_edges,
                                    alpha=0.5,
                                    label=str(item_id)
                                )
                        except Exception:
                            continue

                    axes[1].legend(
                        title="Item ID",
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left"
                    )
                    axes[1].set_xlabel(col)
                    axes[1].set_ylabel("Frequency")
                    axes[1].set_title("Distribution by Item (sampled)")
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()

                    # ======================
                    # 3. 保存
                    # ======================
                    if save:
                        save_path = os.path.join(
                            save_dir,
                            f"distribution_{col}.png"
                        )
                        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

                    plt.close(fig)

                except Exception as e:
                    logger.warning(
                        f"Failed to plot distribution for column '{col}': {e}"
                    )

            return hist_dict

        except Exception as e:
            logger.warning(f"Could not generate distribution plots: {e}")
            return {}


    def _adjust_period(self, period_value: int) -> int:
        """
        【辅助方法】周期调整
        逻辑源自 Characteristics_Extractor.py
        """

        if abs(period_value - 4) <= 1: return 4
        if abs(period_value - 7) <= 1: return 7
        if abs(period_value - 12) <= 2: return 12
        if abs(period_value - 24) <= 3: return 24
        if abs(period_value - 48) <= 1 or ((48 - period_value) <= 4 and (48 - period_value) >= 0): return 48
        if abs(period_value - 52) <= 2: return 52
        if abs(period_value - 96) <= 10: return 96
        if abs(period_value - 144) <= 10: return 144
        if abs(period_value - 168) <= 10: return 168
        if abs(period_value - 336) <= 50: return 336
        if abs(period_value - 672) <= 20: return 672
        if abs(period_value - 720) <= 20: return 720
        if abs(period_value - 1008) <= 100: return 1008
        if abs(period_value - 1440) <= 200: return 1440
        if abs(period_value - 8766) <= 500: return 8766
        if abs(period_value - 10080) <= 500: return 10080
        return period_value

    def _fft_transfer(self, timeseries: np.ndarray, fmin: float = 0.0):
        try:
            # 检查数据长度
            if len(timeseries) < 10:
                logger.debug(f"Timeseries too short for FFT (len={len(timeseries)})")
                return [], []
            
            # 去除NaN
            timeseries = timeseries[~np.isnan(timeseries)]
            if len(timeseries) < 10:
                logger.debug(f"Valid data too short after removing NaN (len={len(timeseries)})")
                return [], []
            
            # 1. 计算FFT
            n = len(timeseries)
            yf = np.fft.fft(timeseries)
            
            # 2. 计算振幅谱
            amplitude = np.abs(yf) / n  # 归一化
            amplitude = amplitude[:n//2] * 2  # 单边谱
            
            # 3. 计算频率
            sampling_rate = 1.0  # 假设单位时间采样
            freqs = np.fft.fftfreq(n, d=1.0)[:n//2]
            
            # 4. 寻找峰值（更宽松的条件）
            from scipy.signal import find_peaks
            
            # 设置峰值检测参数
            height_threshold = np.percentile(amplitude, 70)  # 取前30%的振幅作为阈值
            peaks, properties = find_peaks(
                amplitude, 
                height=height_threshold,
                distance=max(2, n//20)  # 峰值之间最小距离
            )
            
            if len(peaks) == 0:
                # 如果没找到峰值，尝试降低阈值
                height_threshold = np.mean(amplitude)
                peaks, properties = find_peaks(
                    amplitude,
                    height=height_threshold,
                    distance=max(2, n//20)
                )
            
            if len(peaks) == 0:
                logger.debug(f"No significant peaks found in FFT analysis")
                return [], []
            
            # 5. 提取前几个主要周期
            significant_peaks = peaks[:min(5, len(peaks))]  # 最多取5个主要峰值
            
            # 6. 计算周期（周期 = 1 / 频率）
            periods = []
            amplitudes = []
            
            for idx in significant_peaks:
                freq = freqs[idx]
                if freq > 0:  # 忽略零频率（直流分量）
                    period = 1.0 / freq
                    # 只保留合理的周期（不能超过数据长度）
                    if 2 <= period <= n/2:
                        periods.append(period)
                        amplitudes.append(amplitude[idx])
            
            # 按振幅降序排序
            if periods:
                sorted_idx = np.argsort(amplitudes)[::-1]
                periods = np.array(periods)[sorted_idx].tolist()
                amplitudes = np.array(amplitudes)[sorted_idx].tolist()
            
            logger.debug(f"FFT found {len(periods)} periods: {periods[:5]}")
            return periods, amplitudes
            
        except Exception as e:
            logger.error(f"Error during FFT transfer: {str(e)}", exc_info=True)
            return [], []



    def _spatiotemporal_heterogeneity_analysis_KANG(
        self,
        target_column: str,
        max_items: int = 10,
        save: bool = True,
        save_dir: str = "results/plot",
    ):
        """
        带 Logging 和 Visualization 的时空异质性分析 (FFT/STL/ADF-KPSS)
        """
        DEFAULT_PERIODS = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]
        
        if save:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Results will be saved to: {save_dir}")
            # 设置 seaborn 风格，支持中文显示（根据系统字体调整，这里用默认）
            sns.set_theme(style="whitegrid")
            plt.rcParams['axes.unicode_minus'] = False 
        
        # 抽样逻辑
        if hasattr(self, 'num_items') and self.num_items > max_items:
            sampled_items = self.item_ids[:max_items]
            logger.info(f"Sampling {max_items} items from total {self.num_items}.")
        else:
            sampled_items = getattr(self, 'item_ids', [])
            logger.info(f"Processing all {len(sampled_items)} items.")

        results = {
            'trend_analysis': {},
            'seasonal_analysis': {},
            'stationarity_analysis': {}
        }
        
        count_processed = 0
        count_skipped = 0

        for item_id in sampled_items:
            # 数据读取与基础校验
            try:
                item_df = self.loc[item_id]
                if target_column not in item_df.columns:
                    logger.warning(f"Item {item_id}: Column '{target_column}' not found. Skipping.")
                    count_skipped += 1
                    continue
                item_series = item_df[target_column].dropna()
            except Exception as e:
                logger.error(f"Item {item_id}: Error accessing data - {str(e)}")
                count_skipped += 1
                continue

            series_values = item_series.values.astype("float")
            series_length = len(series_values)
            
            # 长度校验
            if series_length < 12:
                logger.warning(f"Item {item_id}: Series too short (len={series_length} < 12). Skipping.")
                count_skipped += 1
                continue

            logger.debug(f"Processing Item {item_id} (len={series_length})...")

            # 1. 平稳性分析 (Stationarity)
            try:
                adf_res = adfuller(series_values, autolag="AIC")
                adf_p = adf_res[1]
                
                kpss_res = kpss(series_values, regression="c", nlags="auto")
                kpss_p = kpss_res[1]
                
                # line 284 logic
                is_stable = (adf_p <= 0.05) or (kpss_p >= 0.05)
                
                logger.debug(f"Item {item_id}: Stationarity - ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}, Stable={is_stable}")
                
            except Exception as e:
                logger.error(f"Item {item_id}: Stationarity test failed - {str(e)}")
                adf_p, kpss_p, is_stable = None, None, False
            
            stat_info = {
                "adf_p_value": adf_p,
                "kpss_p_value": kpss_p,
                "is_stable": is_stable,
                "mean": np.mean(series_values),
                "std": np.std(series_values)
            }
            results['stationarity_analysis'][str(item_id)] = stat_info

            # 2. 周期检测 (FFT)
            fft_periods, fft_amps = self._fft_transfer(series_values, fmin=0)
            
            detected_periods = []
            if len(fft_amps) > 0:
                sorted_idx = np.argsort(fft_amps)[::-1]
                for idx in sorted_idx:
                    detected_periods.append(round(fft_periods[idx]))
            
            # 映射与去重
            candidate_periods = []
            for p in detected_periods:
                adj_p = self._adjust_period(p)
                if adj_p not in candidate_periods and adj_p >= 4:
                    candidate_periods.append(adj_p)
            
            # 最终候选列表
            periods_num = min(len(candidate_periods), 3)
            final_periods_check = candidate_periods[:periods_num] + DEFAULT_PERIODS
            
            final_periods = []
            for p in final_periods_check:
                if p not in final_periods and p >= 4:
                    final_periods.append(p)
            
            logger.info(f"Item {item_id}: Top FFT periods: {detected_periods[:3]} -> Check periods: {final_periods[:5]}...")

            # 3. 趋势与季节强度 (STL)
            season_dict = {}
            threshold_len = max(int(series_length / 3), 12)
            
            for period_val in final_periods:
                if period_val < threshold_len:
                    try:
                        res = STL(item_series, period=period_val).fit()
                        
                        resid = res.resid
                        trend = res.trend
                        seasonal = res.seasonal
                        
                        deseasonal = item_series - seasonal
                        detrend = item_series - trend
                        
                        # 趋势强度
                        var_deseasonal = np.var(deseasonal)
                        t_strength = 0 if var_deseasonal == 0 else max(0, 1 - np.var(resid) / var_deseasonal)
                        
                        # 季节强度
                        var_detrend = np.var(detrend)
                        s_strength = 0 if var_detrend == 0 else max(0, 1 - np.var(resid) / var_detrend)
                        
                        season_dict[s_strength] = {
                            "period": period_val,
                            "seasonal_strength": s_strength,
                            "trend_strength": t_strength,
                            "stl_res": res # 暂存 STL 结果用于绘图 (可选，如果内存够大)
                        }
                    except Exception as e:
                        logger.debug(f"Item {item_id}: STL failed for period {period_val} - {str(e)}")
                        continue
            
            # 提取最佳结果
            best_seasonal_strength = 0
            best_trend_strength = 0
            best_period = 0
            best_stl_res = None
            
            if season_dict:
                max_s_key = max(season_dict.keys())
                best_entry = season_dict[max_s_key]
                
                best_seasonal_strength = best_entry['seasonal_strength']
                best_trend_strength = best_entry['trend_strength']
                best_period = best_entry['period']
                best_stl_res = best_entry.get('stl_res') # 获取对应的 STL 结果对象
            
            has_trend = best_trend_strength >= 0.85
            has_season = best_seasonal_strength >= 0.9

            logger.info(
                f"Item {item_id} Finished: "
                f"Trend={best_trend_strength:.2f} (HasTrend={has_trend}), "
                f"Season={best_seasonal_strength:.2f} (HasSeason={has_season}, Period={best_period})"
            )

            results['trend_analysis'][str(item_id)] = {
                "trend_strength": best_trend_strength,
                "has_significant_trend": has_trend
            }
            
            results['seasonal_analysis'][str(item_id)] = {
                "dominant_period": best_period,
                "seasonal_strength": best_seasonal_strength,
                "has_significant_seasonality": has_season
            }

            # 4. 可视化 (Visualization)
            if save:
                self._plot_analysis_result(
                    item_id=item_id,
                    series=item_series,
                    stat_info=stat_info,
                    best_period=best_period,
                    best_stl_res=best_stl_res,
                    save_dir=save_dir
                )
            
            count_processed += 1

        logger.info(f"Analysis Complete. Processed: {count_processed}, Skipped: {count_skipped}")
        return results


    def _plot_analysis_result(self, item_id, series, stat_info, best_period, best_stl_res, save_dir):
        """
        可视化：平稳性、周期性(FFT)、趋势与季节性(STL)
        """
        try:
            # 创建 3x1 的画布
            fig, axes = plt.subplots(3, 1, figsize=(14, 15), constrained_layout=True)
            fig.suptitle(f"Spatio-Temporal Analysis: Item {item_id}", fontsize=16, fontweight='bold')

            # --- 图1: 原始序列 & 平稳性检测 ---
            ax1 = axes[0]
            ax1.plot(series.values, color='#1f77b4', alpha=0.8, label='Observed')
            ax1.set_title("1. Stationarity Analysis (ADF & KPSS)", fontsize=12, fontweight='bold', loc='left')
            
            # 构建标注文本
            adf_p = stat_info.get('adf_p_value')
            kpss_p = stat_info.get('kpss_p_value')
            is_stable = stat_info.get('is_stable')
            
            stat_text = (
                f"Mean: {stat_info.get('mean', 0):.2f}, Std: {stat_info.get('std', 0):.2f}\n"
                f"ADF p-value: {adf_p:.4f} " + ("(Stationary)" if adf_p and adf_p <= 0.05 else "(Non-Stationary)") + "\n"
                f"KPSS p-value: {kpss_p:.4f} " + ("(Stationary)" if kpss_p and kpss_p >= 0.05 else "(Non-Stationary)") + "\n"
                f"Result: {'STABLE' if is_stable else 'UNSTABLE'}"
            )
            
            # 在图中添加文本框
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax1.text(0.02, 0.95, stat_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            ax1.legend(loc='upper right')
            ax1.set_ylabel("Value")

            # --- 图2: 周期性分析 (FFT Spectrum) ---
            ax2 = axes[1]
            ax2.set_title("2. Periodicity Analysis (FFT Spectrum)", fontsize=12, fontweight='bold', loc='left')
            
            # 重新计算一次完整的 FFT 用于绘图 (因为 _fft_transfer 只返回了 top peaks)
            vals = series.values
            vals = vals[~np.isnan(vals)]
            n = len(vals)
            if n > 0:
                yf = np.fft.fft(vals)
                amplitude = np.abs(yf)[:n//2] * 2 / n
                freqs = np.fft.fftfreq(n, d=1.0)[:n//2]
                
                # 将频率转换为周期 (忽略频率0)
                mask = freqs > 0
                periods = 1.0 / freqs[mask]
                amps = amplitude[mask]
                
                # 过滤掉过长的周期以便绘图更清晰 (可选：限制在数据长度的一半以内)
                valid_mask = periods <= n/2
                plot_periods = periods[valid_mask]
                plot_amps = amps[valid_mask]

                ax2.plot(plot_periods, plot_amps, color='#d62728')
                ax2.set_xlabel("Period (Timesteps)")
                ax2.set_ylabel("Amplitude")
                ax2.set_xscale('log') # 使用对数坐标轴，因为周期跨度可能很大
                ax2.grid(True, which="both", ls="-", alpha=0.5)
                
                # 标注最佳周期
                if best_period > 0:
                    ax2.axvline(x=best_period, color='green', linestyle='--', alpha=0.8, label=f'Best Period: {best_period}')
                    ax2.legend()
            else:
                ax2.text(0.5, 0.5, "Data insufficient for FFT", ha='center')

            # --- 图3: 趋势与季节性分解 (STL) ---
            ax3 = axes[2]
            title_text = f"3. Decomposition (STL) | Best Period: {best_period}"
            if best_stl_res is not None:
                # 绘制 STL 分解结果
                # 为了在一张子图里显示，我们绘制 Trend 和 Seasonal
                ax3.set_title(title_text, fontsize=12, fontweight='bold', loc='left')
                
                # 双轴绘制：左轴 Trend，右轴 Seasonal
                ln1 = ax3.plot(best_stl_res.trend, color='#ff7f0e', label='Trend', linewidth=2)
                ax3.set_ylabel("Trend Level")
                
                ax3_right = ax3.twinx()
                ln2 = ax3_right.plot(best_stl_res.seasonal, color='#2ca02c', alpha=0.6, label='Seasonal', linewidth=1)
                ax3_right.set_ylabel("Seasonal Component")
                
                # 合并图例
                lns = ln1 + ln2
                labs = [l.get_label() for l in lns]
                ax3.legend(lns, labs, loc='upper right')
                
                ax3.grid(True, alpha=0.3)
            else:
                ax3.set_title(title_text + " (Failed or No Period)", fontsize=12, fontweight='bold', loc='left')
                ax3.text(0.5, 0.5, "STL Decomposition Not Available", ha='center', transform=ax3.transAxes)

            # 保存图片
            save_path = os.path.join(save_dir, f"analysis_{item_id}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed for item {item_id}: {str(e)}", exc_info=True)
        finally:
            # 关闭图形以释放内存
            plt.close(fig)

    
    @classmethod
    def _validate_data_frame(cls, df: pd.DataFrame):
        """Validate that a pd.DataFrame with ITEMID and TIMESTAMP columns can be converted to TimeSeriesDataFrame"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(df)}")
        if cls.ITEMID not in df.columns:
            raise ValueError(f"data must have a `{cls.ITEMID}` column")
        if cls.TIMESTAMP not in df.columns:
            raise ValueError(f"data must have a `{cls.TIMESTAMP}` column")
        if df[cls.ITEMID].isnull().any():
            raise ValueError(f"`{cls.ITEMID}` column can not have nan")
        if df[cls.TIMESTAMP].isnull().any():
            raise ValueError(f"`{cls.TIMESTAMP}` column can not have nan")
        if not pd.api.types.is_datetime64_dtype(df[cls.TIMESTAMP]):
            raise ValueError(f"for {cls.TIMESTAMP}, the only pandas dtype allowed is `datetime64`.")
        item_id_column = df[cls.ITEMID]
        if not (pd.api.types.is_integer_dtype(item_id_column) or pd.api.types.is_string_dtype(item_id_column)):
            raise ValueError(f"all entries in column `{cls.ITEMID}` must be of integer or string dtype")

    @classmethod
    def _validate_iterable(cls, data: Iterable):
        if not isinstance(data, Iterable):
            raise ValueError("data must be of type Iterable.")

        first = next(iter(data), None)
        if first is None:
            raise ValueError("data has no time-series.")

        for i, ts in enumerate(itertools.chain([first], data)):
            if not isinstance(ts, dict):
                raise ValueError(f"{i}'th time-series in data must be a dict, got{type(ts)}")
            if not ("target" in ts and "start" in ts):
                raise ValueError(f"{i}'th time-series in data must have 'target' and 'start', got{ts.keys()}")
            if not isinstance(ts["start"], pd.Period):
                raise ValueError(f"{i}'th time-series must have a pandas Period as 'start', got {ts['start']}")

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: str | None = None,
        timestamp_column: str | None = None,
        static_features_df: pd.DataFrame | None = None,
    ) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame with 'item_id' and 'timestamp' as columns. For example::

                   item_id  timestamp  target
                0        0 2019-01-01       0
                1        0 2019-01-02       1
                2        0 2019-01-03       2
                3        1 2019-01-01       3
                4        1 2019-01-02       4
                5        1 2019-01-03       5
                6        2 2019-01-01       6
                7        2 2019-01-02       7
                8        2 2019-01-03       8
        id_column : str, optional
            Name of the 'item_id' column if column name is different
        timestamp_column : str, optional
            Name of the 'timestamp' column if column name is different
        static_features_df : pd.DataFrame, optional
            A pd.DataFrame with 'item_id' column that contains the static features for each time series. For example::

                   item_id feat_1   feat_2
                0        0 foo         0.5
                1        1 foo         2.2
                2        2 bar         0.1

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A dataframe in TimeSeriesDataFrame format.
        """
        return cls(df, static_features=static_features_df, id_column=id_column, timestamp_column=timestamp_column)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        id_column: str | None = None,
        timestamp_column: str | None = None,
        static_features_path: str | Path | None = None,
    ) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from a CSV or Parquet file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a local or remote (e.g., S3) file containing the time series data in CSV or Parquet format.
            Example file contents::

                item_id,timestamp,target
                0,2019-01-01,0
                0,2019-01-02,1
                0,2019-01-03,2
                1,2019-01-01,3
                1,2019-01-02,4
                1,2019-01-03,5
                2,2019-01-01,6
                2,2019-01-02,7
                2,2019-01-03,8

        id_column : str, optional
            Name of the 'item_id' column if column name is different
        timestamp_column : str, optional
            Name of the 'timestamp' column if column name is different
        static_features_path : str or pathlib.Path, optional
            Path to a local or remote (e.g., S3) file containing static features in CSV or Parquet format.
            Example file contents::

                item_id,feat_1,feat_2
                0,foo,0.5
                1,foo,2.2
                2,bar,0.1

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A dataframe in TimeSeriesDataFrame format.
        """
        return cls(path, static_features=static_features_path, id_column=id_column, timestamp_column=timestamp_column)

    @classmethod
    def from_iterable_dataset(cls, iterable_dataset: Iterable, num_cpus: int = -1) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from an Iterable of dictionaries each of which
        represent a single time series.

        This function also offers compatibility with GluonTS `ListDataset format <https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html#gluonts.dataset.common.ListDataset>`_.

        Parameters
        ----------
        iterable_dataset: Iterable
            An iterator over dictionaries, each with a ``target`` field specifying the value of the
            (univariate) time series, and a ``start`` field with the starting time as a pandas Period .
            Example::

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Period("01-01-2019", freq='D')}
                ]
        num_cpus : int, default = -1
            Number of CPU cores used to process the iterable dataset in parallel. Set to -1 to use all cores.

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A dataframe in TimeSeriesDataFrame format.
        """
        return cls(iterable_dataset, num_cpus=num_cpus)

    @property
    def item_ids(self) -> pd.Index:
        """List of unique time series IDs contained in the data set."""
        return self.index.unique(level=self.ITEMID)

    @classmethod
    def _construct_static_features(
        cls,
        static_features: pd.DataFrame | str | Path,
        id_column: str | None = None,
    ) -> pd.DataFrame:
        if isinstance(static_features, (str, Path)):
            static_features = load_pd.load(str(static_features))
        if not isinstance(static_features, pd.DataFrame):
            raise ValueError(
                f"static_features must be a pd.DataFrame, string or Path (received {type(static_features)})"
            )

        if id_column is not None:
            assert id_column in static_features.columns, f"Column '{id_column}' not found in static_features!"
            if id_column != cls.ITEMID and cls.ITEMID in static_features.columns:
                logger.warning(
                    f"Renaming existing column '{cls.ITEMID}' -> '__{cls.ITEMID}' to avoid name collisions."
                )
                static_features.rename(columns={cls.ITEMID: "__" + cls.ITEMID}, inplace=True)
            static_features.rename(columns={id_column: cls.ITEMID}, inplace=True)
        return static_features

    @property
    def static_features(self):
        return self._static_features

    @static_features.setter
    def static_features(self, value: pd.DataFrame | None):
        # if the current item index is not a multiindex, then we are dealing with a single
        # item slice. this should only happen when the user explicitly requests only a
        # single item or during `slice_by_timestep`. In this case we do not set static features
        if not isinstance(self.index, pd.MultiIndex):
            return

        if value is not None:
            if isinstance(value, pd.Series):
                value = value.to_frame()
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"static_features must be a pandas DataFrame (received object of type {type(value)})")
            if isinstance(value.index, pd.MultiIndex):
                raise ValueError("static_features cannot have a MultiIndex")

            # Avoid modifying static features inplace
            value = value.copy()
            if self.ITEMID in value.columns and value.index.name != self.ITEMID:
                value = value.set_index(self.ITEMID)
            if value.index.name != self.ITEMID:
                value.index.rename(self.ITEMID, inplace=True)
            missing_item_ids = self.item_ids.difference(value.index)
            if len(missing_item_ids) > 0:
                raise ValueError(
                    "Following item_ids are missing from the index of static_features: "
                    f"{reprlib.repr(missing_item_ids.to_list())}"
                )
            # if provided static features are a strict superset of the item index, we take a subset to ensure consistency
            if len(value.index.difference(self.item_ids)) > 0:
                value = value.reindex(self.item_ids)

        self._static_features = value

    def infer_frequency(self, num_items: int | None = None, raise_if_irregular: bool = False) -> str:
        """Infer the time series frequency based on the timestamps of the observations.

        Parameters
        ----------
        num_items : int or None, default = None
            Number of items (individual time series) randomly selected to infer the frequency. Lower values speed up
            the method, but increase the chance that some items with invalid frequency are missed by subsampling.

            If set to ``None``, all items will be used for inferring the frequency.
        raise_if_irregular : bool, default = False
            If True, an exception will be raised if some items have an irregular frequency, or if different items have
            different frequencies.

        Returns
        -------
        freq : str
            If all time series have a regular frequency, returns a pandas-compatible `frequency alias <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.

            If some items have an irregular frequency or if different items have different frequencies, returns string
            ``IRREG``.
        """
        ts_df = self
        if num_items is not None and ts_df.num_items > num_items:
            items_subset = ts_df.item_ids.to_series().sample(n=num_items, random_state=123)
            ts_df = ts_df.loc[items_subset]

        if not ts_df.index.is_monotonic_increasing:
            ts_df = ts_df.sort_index()

        indptr = ts_df.get_indptr()
        item_ids = ts_df.item_ids
        timestamps = ts_df.index.get_level_values(level=1)
        candidate_freq = ts_df.index.levels[1].freq

        frequencies = []
        irregular_items = []
        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i + 1]
            item_timestamps = timestamps[start:end]
            inferred_freq = item_timestamps.inferred_freq

            # Fallback option: maybe original index has a `freq` attribute that pandas fails to infer (e.g., 'SME')
            if inferred_freq is None and candidate_freq is not None:
                try:
                    # If this line does not raise an exception, then candidate_freq is a compatible frequency
                    item_timestamps.freq = candidate_freq
                except ValueError:
                    inferred_freq = None
                else:
                    inferred_freq = candidate_freq.freqstr

            if inferred_freq is None:
                irregular_items.append(item_ids[i])
            else:
                frequencies.append(inferred_freq)

        unique_freqs = list(set(frequencies))
        if len(unique_freqs) != 1 or len(irregular_items) > 0:
            if raise_if_irregular:
                if irregular_items:
                    raise ValueError(
                        f"Cannot infer frequency. Items with irregular frequency: {reprlib.repr(irregular_items)}"
                    )
                else:
                    raise ValueError(f"Cannot infer frequency. Multiple frequencies detected: {unique_freqs}")
            else:
                return self.IRREGULAR_TIME_INDEX_FREQSTR
        else:
            return pd.tseries.frequencies.to_offset(unique_freqs[0]).freqstr

    @property
    def freq(self):
        """Inferred pandas-compatible frequency of the timestamps in the dataframe.

        Computed using a random subset of the time series for speed. This may sometimes result in incorrectly inferred
        values. For reliable results, use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.infer_frequency`.
        """
        inferred_freq = self.infer_frequency(num_items=50)
        return None if inferred_freq == self.IRREGULAR_TIME_INDEX_FREQSTR else inferred_freq

    @property
    def num_items(self):
        """Number of items (time series) in the data set."""
        return len(self.item_ids)

    def num_timesteps_per_item(self) -> pd.Series:
        """Number of observations in each time series in the dataframe.

        Returns a ``pandas.Series`` with ``item_id`` as index and number of observations per item as values.
        """
        counts = pd.Series(self.index.codes[0]).value_counts(sort=False)
        counts.index = self.index.levels[0][counts.index]
        return counts

    def copy(self: TimeSeriesDataFrame, deep: bool = True) -> TimeSeriesDataFrame: # type: ignore
        """Make a copy of the TimeSeriesDataFrame.

        When ``deep=True`` (default), a new object will be created with a copy of the calling object's data and
        indices. Modifications to the data or indices of the copy will not be reflected in the original object.

        When ``deep=False``, a new object will be created without copying the calling object's data or index (only
        references to the data and index are copied). Any changes to the data of the original will be reflected in the
        shallow copy (and vice versa).

        For more details, see `pandas documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html>`_.
        """
        obj = super().copy(deep=deep)

        # also perform a deep copy for static features
        if deep:
            for k in obj._metadata:
                setattr(obj, k, copy.deepcopy(getattr(obj, k)))
        return obj

    def __finalize__(  # noqa # type: ignore
        self: TimeSeriesDataFrame, other, method: str | None = None, **kwargs
    ) -> TimeSeriesDataFrame:
        super().__finalize__(other=other, method=method, **kwargs)
        # when finalizing the copy/slice operation, we use the property setter to stay consistent
        # with the item index
        if hasattr(other, "_static_features"):
            self.static_features = other._static_features
        return self

    def split_by_time(self, cutoff_time: pd.Timestamp) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split dataframe to two different ``TimeSeriesDataFrame`` s before and after a certain ``cutoff_time``.

        Parameters
        ----------
        cutoff_time: pd.Timestamp
            The time to split the current dataframe into two dataframes.

        Returns
        -------
        data_before: TimeSeriesDataFrame
            Data frame containing time series before the ``cutoff_time`` (exclude ``cutoff_time``).
        data_after: TimeSeriesDataFrame
            Data frame containing time series after the ``cutoff_time`` (include ``cutoff_time``).
        """

        nanosecond_before_cutoff = cutoff_time - pd.Timedelta(nanoseconds=1) # type: ignore
        data_before = self.loc[(slice(None), slice(None, nanosecond_before_cutoff)), :]
        data_after = self.loc[(slice(None), slice(cutoff_time, None)), :]
        before = TimeSeriesDataFrame(data_before, static_features=self.static_features)
        after = TimeSeriesDataFrame(data_after, static_features=self.static_features)
        return before, after

    def slice_by_timestep(self, start_index: int | None = None, end_index: int | None = None) -> TimeSeriesDataFrame:
        """Select a subsequence from each time series between start (inclusive) and end (exclusive) indices.

        This operation is equivalent to selecting a slice ``[start_index : end_index]`` from each time series, and then
        combining these slices into a new ``TimeSeriesDataFrame``. See examples below.

        It is recommended to sort the index with ``ts_df.sort_index()`` before calling this method to take advantage of
        a fast optimized algorithm.

        Parameters
        ----------
        start_index : int or None
            Start index (inclusive) of the slice for each time series.
            Negative values are counted from the end of each time series.
            When set to None, the slice starts from the beginning of each time series.
        end_index : int or None
            End index (exclusive) of the slice for each time series.
            Negative values are counted from the end of each time series.
            When set to None, the slice includes the end of each time series.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            A new time series dataframe containing entries of the original time series between start and end indices.

        Examples
        --------
        >>> ts_df
                            target
        item_id timestamp
        0       2019-01-01       0
                2019-01-02       1
                2019-01-03       2
        1       2019-01-02       3
                2019-01-03       4
                2019-01-04       5
        2       2019-01-03       6
                2019-01-04       7
                2019-01-05       8

        Select the first entry of each time series

        >>> df.slice_by_timestep(0, 1)
                            target
        item_id timestamp
        0       2019-01-01       0
        1       2019-01-02       3
        2       2019-01-03       6

        Select the last 2 entries of each time series

        >>> df.slice_by_timestep(-2, None)
                            target
        item_id timestamp
        0       2019-01-02       1
                2019-01-03       2
        1       2019-01-03       4
                2019-01-04       5
        2       2019-01-04       7
                2019-01-05       8

        Select all except the last entry of each time series

        >>> df.slice_by_timestep(None, -1)
                            target
        item_id timestamp
        0       2019-01-01       0
                2019-01-02       1
        1       2019-01-02       3
                2019-01-03       4
        2       2019-01-03       6
                2019-01-04       7

        Copy the entire dataframe

        >>> df.slice_by_timestep(None, None)
                            target
        item_id timestamp
        0       2019-01-01       0
                2019-01-02       1
                2019-01-03       2
        1       2019-01-02       3
                2019-01-03       4
                2019-01-04       5
        2       2019-01-03       6
                2019-01-04       7
                2019-01-05       8

        """
        if start_index is not None and not isinstance(start_index, int):
            raise ValueError(f"start_index must be of type int or None (got {type(start_index)})")
        if end_index is not None and not isinstance(end_index, int):
            raise ValueError(f"end_index must be of type int or None (got {type(end_index)})")

        if start_index is None and end_index is None:
            # Return a copy to avoid in-place modification.
            # self.copy() is much faster than self.loc[ones(len(self), dtype=bool)]
            return self.copy()

        if self.index.is_monotonic_increasing:
            # Use a fast optimized algorithm if the index is sorted
            indptr = self.get_indptr()
            lengths = np.diff(indptr)
            starts = indptr[:-1]

            slice_start = (
                np.zeros_like(lengths)
                if start_index is None
                else np.clip(np.where(start_index >= 0, start_index, lengths + start_index), 0, lengths)
            )
            slice_end = (
                lengths.copy()
                if end_index is None
                else np.clip(np.where(end_index >= 0, end_index, lengths + end_index), 0, lengths)
            )

            # Filter out invalid slices where start >= end
            valid_slices = slice_start < slice_end
            if not np.any(valid_slices):
                # Return empty dataframe with same structure
                return self.loc[np.zeros(len(self), dtype=bool)]

            starts = starts[valid_slices]
            slice_start = slice_start[valid_slices]
            slice_end = slice_end[valid_slices]

            # We put 1 at the slice_start index for each item and -1 at the slice_end index for each item.
            # After we apply cumsum we get the indicator mask selecting values between slice_start and slice_end
            # cumsum([0, 0, 1, 0, 0, -1, 0]) -> [0, 0, 1, 1, 1, 0, 0]
            # We need array of size len(self) + 1 in case events[starts + slice_end] tries to access position len(self)
            events = np.zeros(len(self) + 1, dtype=np.int8)
            events[starts + slice_start] += 1
            events[starts + slice_end] -= 1
            mask = np.cumsum(events)[:-1].astype(bool)
            # loc[mask] returns a view of the original data - modifying it will produce a SettingWithCopyWarning
            return self.loc[mask]
        else:
            # Fall back to a slow groupby operation
            result = self.groupby(level=self.ITEMID, sort=False, as_index=False).nth(slice(start_index, end_index))
            result.static_features = self.static_features
            return result

    def slice_by_time(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> TimeSeriesDataFrame:
        """Select a subsequence from each time series between start (inclusive) and end (exclusive) timestamps.

        Parameters
        ----------
        start_time: pd.Timestamp
            Start time (inclusive) of the slice for each time series.
        end_time: pd.Timestamp
            End time (exclusive) of the slice for each time series.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            A new time series dataframe containing entries of the original time series between start and end timestamps.
        """

        if end_time < start_time:
            raise ValueError(f"end_time {end_time} is earlier than start_time {start_time}")

        nanosecond_before_end_time = end_time - pd.Timedelta(nanoseconds=1)
        return TimeSeriesDataFrame(
            self.loc[(slice(None), slice(start_time, nanosecond_before_end_time)), :],
            static_features=self.static_features,
        )

    @classmethod
    def from_pickle(cls, filepath_or_buffer: Any) -> TimeSeriesDataFrame:
        """Convenience method to read pickled time series dataframes. If the read pickle
        file refers to a plain pandas DataFrame, it will be cast to a TimeSeriesDataFrame.

        Parameters
        ----------
        filepath_or_buffer: Any
            Filename provided as a string or an ``IOBuffer`` containing the pickled object.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            The pickled time series dataframe.
        """
        try:
            data = pd.read_pickle(filepath_or_buffer)
            return data if isinstance(data, cls) else cls(data)
        except Exception as err:  # noqa
            raise IOError(f"Could not load pickled data set due to error: {str(err)}")

    def fill_missing_values(self, method: str = "auto", value: float = 0.0) -> TimeSeriesDataFrame:
        """Fill missing values represented by NaN.

        .. note::
            This method assumes that the index of the TimeSeriesDataFrame is sorted by [item_id, timestamp].

            If the index is not sorted, this method will log a warning and may produce an incorrect result.

        Parameters
        ----------
        method : str, default = "auto"
            Method used to impute missing values.

            - ``"auto"`` - first forward fill (to fill the in-between and trailing NaNs), then backward fill (to fill the leading NaNs)
            - ``"ffill"`` or ``"pad"`` - propagate last valid observation forward. Note: missing values at the start of the time series are not filled.
            - ``"bfill"`` or ``"backfill"`` - use next valid observation to fill gap. Note: this may result in information leakage; missing values at the end of the time series are not filled.
            - ``"constant"`` - replace NaNs with the given constant ``value``.
            - ``"interpolate"`` - fill NaN values using linear interpolation. Note: this may result in information leakage.
        value : float, default = 0.0
            Value used by the "constant" imputation method.

        Examples
        --------
        >>> ts_df
                            target
        item_id timestamp
        0       2019-01-01     NaN
                2019-01-02     NaN
                2019-01-03     1.0
                2019-01-04     NaN
                2019-01-05     NaN
                2019-01-06     2.0
                2019-01-07     NaN
        1       2019-02-04     NaN
                2019-02-05     3.0
                2019-02-06     NaN
                2019-02-07     4.0

        >>> ts_df.fill_missing_values(method="auto")
                            target
        item_id timestamp
        0       2019-01-01     1.0
                2019-01-02     1.0
                2019-01-03     1.0
                2019-01-04     1.0
                2019-01-05     1.0
                2019-01-06     2.0
                2019-01-07     2.0
        1       2019-02-04     3.0
                2019-02-05     3.0
                2019-02-06     3.0
                2019-02-07     4.0

        """
        # Convert to pd.DataFrame for faster processing
        df = pd.DataFrame(self)

        # Skip filling if there are no NaNs
        if not df.isna().any(axis=None):
            return self

        if not self.index.is_monotonic_increasing:
            logger.warning(
                "Trying to fill missing values in an unsorted dataframe. "
                "It is highly recommended to call `ts_df.sort_index()` before calling `ts_df.fill_missing_values()`"
            )

        grouped_df = df.groupby(level=self.ITEMID, sort=False, group_keys=False)
        if method == "auto":
            filled_df = grouped_df.ffill()
            # If necessary, fill missing values at the start of each time series with bfill
            if filled_df.isna().any(axis=None):
                filled_df = filled_df.groupby(level=self.ITEMID, sort=False, group_keys=False).bfill()
        elif method in ["ffill", "pad"]:
            filled_df = grouped_df.ffill()
        elif method in ["bfill", "backfill"]:
            filled_df = grouped_df.bfill()
        elif method == "constant":
            filled_df = self.fillna(value=value)
        elif method == "interpolate":
            filled_df = grouped_df.apply(lambda ts: ts.interpolate())
        else:
            raise ValueError(
                "Invalid fill method. Expecting one of "
                "{'auto', 'ffill', 'pad', 'bfill', 'backfill', 'constant', 'interpolate'}. "
                f"Got {method}"
            )
        return TimeSeriesDataFrame(filled_df, static_features=self.static_features)

    def dropna(self, how: str = "any") -> TimeSeriesDataFrame:  # type: ignore[override]
        """Drop rows containing NaNs.

        Parameters
        ----------
        how : {"any", "all"}, default = "any"
            Determine if row or column is removed from TimeSeriesDataFrame, when we have at least one NaN or all NaN.

            - "any" : If any NaN values are present, drop that row or column.
            - "all" : If all values are NaN, drop that row or column.
        """
        # We need to cast to a DataFrame first. Calling self.dropna() results in an exception because self.T
        # (used inside dropna) is not supported for TimeSeriesDataFrame
        dropped_df = pd.DataFrame(self).dropna(how=how)
        return TimeSeriesDataFrame(dropped_df, static_features=self.static_features)

    # added for static type checker compatibility
    def assign(self, **kwargs) -> TimeSeriesDataFrame:
        """Assign new columns to the time series dataframe. See :meth:`pandas.DataFrame.assign` for details."""
        return super().assign(**kwargs)  # type: ignore

    # added for static type checker compatibility
    def sort_index(self, *args, **kwargs) -> TimeSeriesDataFrame:
        return super().sort_index(*args, **kwargs)  # type: ignore

    def get_model_inputs_for_scoring(
        self, prediction_length: int, known_covariates_names: list[str] | None = None
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame | None]:
        """Prepare model inputs necessary to predict the last ``prediction_length`` time steps of each time series in the dataset.

        Parameters
        ----------
        prediction_length : int
            The forecast horizon, i.e., How many time steps into the future must be predicted.
        known_covariates_names : list[str], optional
            Names of the dataframe columns that contain covariates known in the future.
            See ``known_covariates_names`` of :class:`~autogluon.timeseries.TimeSeriesPredictor` for more details.

        Returns
        -------
        past_data : TimeSeriesDataFrame
            Data, where the last ``prediction_length`` time steps have been removed from the end of each time series.
        known_covariates : TimeSeriesDataFrame or None
            If ``known_covariates_names`` was provided, dataframe with the values of the known covariates during the
            forecast horizon. Otherwise, ``None``.
        """
        past_data = self.slice_by_timestep(None, -prediction_length)
        if known_covariates_names is not None and len(known_covariates_names) > 0:
            future_data = self.slice_by_timestep(-prediction_length, None)
            known_covariates = future_data[known_covariates_names]
        else:
            known_covariates = None
        return past_data, known_covariates

    def train_test_split(
        self,
        prediction_length: int,
        end_index: int | None = None,
        suffix: str | None = None,
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Generate a train/test split from the given dataset.

        This method can be used to generate splits for multi-window backtesting.

        .. note::
            This method automatically sorts the TimeSeriesDataFrame by [item_id, timestamp].

        Parameters
        ----------
        prediction_length : int
            Number of time steps in a single evaluation window.
        end_index : int, optional
            If given, all time series will be shortened up to ``end_idx`` before the train/test splitting. In other
            words, test data will include the slice ``[:end_index]`` of each time series, and train data will include
            the slice ``[:end_index - prediction_length]``.
        suffix : str, optional
            Suffix appended to all entries in the ``item_id`` index level.

        Returns
        -------
        train_data : TimeSeriesDataFrame
            Train portion of the data. Contains the slice ``[:-prediction_length]`` of each time series in ``test_data``.
        test_data : TimeSeriesDataFrame
            Test portion of the data. Contains the slice ``[:end_idx]`` of each time series in the original dataset.
        """
        df = self
        if not df.index.is_monotonic_increasing:
            logger.warning("Sorting the dataframe index before generating the train/test split.")
            df = df.sort_index()
        test_data = df.slice_by_timestep(None, end_index)
        train_data = test_data.slice_by_timestep(None, -prediction_length)

        if suffix is not None:
            for data in [train_data, test_data]:
                new_item_id = data.index.levels[0].astype(str) + suffix
                data.index = data.index.set_levels(levels=new_item_id, level=0)
                if data.static_features is not None:
                    data.static_features.index = data.static_features.index.astype(str)
                    data.static_features.index += suffix
        return train_data, test_data

    def convert_frequency(
        self,
        freq: str | pd.DateOffset,
        agg_numeric: str = "mean",
        agg_categorical: str = "first",
        num_cpus: int = -1,
        chunk_size: int = 100,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Convert each time series in the dataframe to the given frequency.

        This method is useful for two purposes:

        1. Converting an irregularly-sampled time series to a regular time index.
        2. Aggregating time series data by downsampling (e.g., convert daily sales into weekly sales)

        Standard ``df.groupby(...).resample(...)`` can be extremely slow for large datasets, so we parallelize this
        operation across multiple CPU cores.

        Parameters
        ----------
        freq : str | pd.DateOffset
            Frequency to which the data should be converted. See `pandas frequency aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            for supported values.
        agg_numeric : {"max", "min", "sum", "mean", "median", "first", "last"}, default = "mean"
            Aggregation method applied to numeric columns.
        agg_categorical : {"first", "last"}, default = "first"
            Aggregation method applied to categorical columns.
        num_cpus : int, default = -1
            Number of CPU cores used when resampling in parallel. Set to -1 to use all cores.
        chunk_size : int, default = 100
            Number of time series in a chunk assigned to each parallel worker.
        **kwargs
            Additional keywords arguments that will be passed to ``pandas.DataFrameGroupBy.resample``.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            A new time series dataframe with time series resampled at the new frequency. Output may contain missing
            values represented by ``NaN`` if original data does not have information for the given period.

        Examples
        --------
        Convert irregularly-sampled time series data to a regular index

        >>> ts_df
                            target
        item_id timestamp
        0       2019-01-01     NaN
                2019-01-03     1.0
                2019-01-06     2.0
                2019-01-07     NaN
        1       2019-02-04     3.0
                2019-02-07     4.0
        >>> ts_df.convert_frequency(freq="D")
                            target
        item_id timestamp
        0       2019-01-01     NaN
                2019-01-02     NaN
                2019-01-03     1.0
                2019-01-04     NaN
                2019-01-05     NaN
                2019-01-06     2.0
                2019-01-07     NaN
        1       2019-02-04     3.0
                2019-02-05     NaN
                2019-02-06     NaN
                2019-02-07     4.0

        Downsample quarterly data to yearly frequency

        >>> ts_df
                            target
        item_id timestamp
        0       2020-03-31     1.0
                2020-06-30     2.0
                2020-09-30     3.0
                2020-12-31     4.0
                2021-03-31     5.0
                2021-06-30     6.0
                2021-09-30     7.0
                2021-12-31     8.0
        >>> ts_df.convert_frequency("YE")
                            target
        item_id timestamp
        0       2020-12-31     2.5
                2021-12-31     6.5
        >>> ts_df.convert_frequency("YE", agg_numeric="sum")
                            target
        item_id timestamp
        0       2020-12-31    10.0
                2021-12-31    26.0
        """
        offset = pd.tseries.frequencies.to_offset(freq)

        # We need to aggregate categorical columns separately because .agg("mean") deletes all non-numeric columns
        aggregation = {}
        for col in self.columns:
            if pd.api.types.is_numeric_dtype(self.dtypes[col]):
                aggregation[col] = agg_numeric
            else:
                aggregation[col] = agg_categorical

        def split_into_chunks(iterable: Iterable, size: int) -> Iterable[Iterable]:
            # Based on https://stackoverflow.com/a/22045226/5497447
            iterable = iter(iterable)
            return iter(lambda: tuple(islice(iterable, size)), ())

        def resample_chunk(chunk: Iterable[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
            resampled_dfs = []
            for item_id, df in chunk:
                resampled_df = df.resample(offset, level=self.TIMESTAMP, **kwargs).agg(aggregation)
                resampled_dfs.append(pd.concat({item_id: resampled_df}, names=[self.ITEMID]))
            return pd.concat(resampled_dfs)

        # Resampling time for 1 item < overhead time for a single parallel job. Therefore, we group items into chunks
        # so that the speedup from parallelization isn't dominated by the communication costs.
        df = pd.DataFrame(self)
        # Make sure that timestamp index has dtype 'datetime64[ns]', otherwise index may contain NaT values.
        # See https://github.com/autogluon/autogluon/issues/4917
        df.index = df.index.set_levels(df.index.levels[1].astype("datetime64[ns]"), level=self.TIMESTAMP)
        chunks = split_into_chunks(df.groupby(level=self.ITEMID, sort=False), chunk_size)
        resampled_chunks = Parallel(n_jobs=num_cpus)(delayed(resample_chunk)(chunk) for chunk in chunks)
        resampled_df = TimeSeriesDataFrame(pd.concat(resampled_chunks))
        resampled_df.static_features = self.static_features
        return resampled_df

    def to_data_frame(self) -> pd.DataFrame:
        """Convert ``TimeSeriesDataFrame`` to a ``pandas.DataFrame``"""
        return pd.DataFrame(self)

    def get_indptr(self) -> np.ndarray:
        """[Advanced] Get a numpy array of shape [num_items + 1] that points to the start and end of each time series.

        This method assumes that the TimeSeriesDataFrame is sorted by [item_id, timestamp].
        """
        return np.concatenate([[0], np.cumsum(self.num_timesteps_per_item().to_numpy())]).astype(np.int32)

    # inline typing stubs for various overridden methods
    if TYPE_CHECKING:

        def query(  # type: ignore
            self, expr: str, *, inplace: bool = False, **kwargs
        ) -> Self: ...

        def reindex(*args, **kwargs) -> Self: ...  # type: ignore

        @overload
        def __new__(cls, data: pd.DataFrame, static_features: pd.DataFrame | None = None) -> Self: ...  # type: ignore
        @overload
        def __new__(
            cls,
            data: pd.DataFrame | str | Path | Iterable,
            static_features: pd.DataFrame | str | Path | None = None,
            id_column: str | None = None,
            timestamp_column: str | None = None,
            num_cpus: int = -1,
            *args,
            **kwargs,
        ) -> Self:
            """This overload is needed since in pandas, during type checking, the default constructor resolves to __new__"""
            ...

        @overload
        def __getitem__(self, items: list[str]) -> Self: ...  # type: ignore
        @overload
        def __getitem__(self, item: str) -> pd.Series: ...  # type: ignore


# TODO: remove with v2.0
# module-level constants kept for backward compatibility.
ITEMID = TimeSeriesDataFrame.ITEMID
TIMESTAMP = TimeSeriesDataFrame.TIMESTAMP
