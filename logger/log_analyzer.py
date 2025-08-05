"""
RAG Log Analysis Utilities - Complete Fixed Version
Provides tools to analyze and visualize RAG pipeline logs for performance optimization.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import argparse

# Optional imports with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Some analysis features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Dashboard generation will be disabled.")

# Import the RAG logger (assuming it's in the same directory)
try:
    from rag_pipeline_logger import RAGPipelineLogger
except ImportError:
    # If the import fails, we'll create a minimal mock for testing
    print("Warning: RAGPipelineLogger not found. Using mock for testing.")
    
    class RAGPipelineLogger:
        def __init__(self, config_path=None):
            self.base_log_dir = 'logs'
            self.temp_log_dir = 'temp_logs'
        
        def get_logger(self, name):
            import logging
            return logging.getLogger(name)
        
        def get_temp_logs_for_analysis(self, request_id=None, log_type=None):
            return []
        
        def _get_temp_log_path(self, log_type, filename):
            return os.path.join(self.temp_log_dir, log_type, filename)


class RAGLogAnalyzer:
    """Comprehensive analysis tools for RAG pipeline logs."""
    
    def __init__(self, logger: RAGPipelineLogger):
        self.logger = logger
        self.analyzer_logger = logger.get_logger('log_analyzer')
    
    def load_all_metrics(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Load all metrics data. Returns list if pandas not available, DataFrame if available."""
        self.analyzer_logger.info(f"Loading metrics data for last {days_back} days")
        
        # Get all temp logs
        temp_logs = self.logger.get_temp_logs_for_analysis(log_type='metrics')
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        metrics_data = []
        for log in temp_logs:
            try:
                log_data = log['data']
                log_time = datetime.fromisoformat(log_data['timestamp'])
                
                if log_time >= cutoff_date:
                    metrics_data.append(log_data)
            except (KeyError, ValueError) as e:
                self.analyzer_logger.warning(f"Skipping invalid log entry: {e}")
                continue
        
        if not metrics_data:
            self.analyzer_logger.warning("No metrics data found")
            return [] if not HAS_PANDAS else pd.DataFrame()
        
        self.analyzer_logger.info(f"Loaded {len(metrics_data)} metrics records")
        
        # Return pandas DataFrame if available, otherwise return list
        if HAS_PANDAS:
            df = pd.DataFrame(metrics_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return metrics_data
    
    def analyze_performance_patterns(self, data) -> Dict[str, Any]:
        """Analyze performance patterns from metrics data."""
        if (HAS_PANDAS and isinstance(data, pd.DataFrame) and data.empty) or \
           (not HAS_PANDAS and isinstance(data, list) and not data):
            return {}
        
        # Handle both pandas DataFrame and list of dictionaries
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            return self._analyze_with_pandas(data)
        else:
            return self._analyze_with_basic_python(data)
    
    def _analyze_with_pandas(self, df) -> Dict[str, Any]:
        """Analyze performance patterns using pandas."""
        analysis = {
            'summary_stats': {
                'total_requests': len(df),
                'success_rate': df['success'].mean() if 'success' in df.columns else 0,
                'avg_total_time': df['total_time'].mean() if 'total_time' in df.columns else 0,
                'avg_retrieval_time': df['retrieval_time'].mean() if 'retrieval_time' in df.columns else 0,
                'avg_generation_time': df['generation_time'].mean() if 'generation_time' in df.columns else 0,
                'avg_documents_retrieved': df['documents_retrieved'].mean() if 'documents_retrieved' in df.columns else 0
            }
        }
        
        # Time-based analysis
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            
            analysis['hourly_patterns'] = {
                'requests_by_hour': df.groupby('hour').size().to_dict(),
                'avg_response_time_by_hour': df.groupby('hour')['total_time'].mean().to_dict() if 'total_time' in df.columns else {}
            }
            
            analysis['daily_patterns'] = {
                'requests_by_day': df.groupby('day_of_week').size().to_dict(),
                'avg_response_time_by_day': df.groupby('day_of_week')['total_time'].mean().to_dict() if 'total_time' in df.columns else {}
            }
        
        # Performance distribution analysis
        if 'total_time' in df.columns:
            analysis['performance_distribution'] = {
                'percentiles': {
                    '50th': df['total_time'].quantile(0.5),
                    '90th': df['total_time'].quantile(0.9),
                    '95th': df['total_time'].quantile(0.95),
                    '99th': df['total_time'].quantile(0.99)
                },
                'slow_requests_count': len(df[df['total_time'] > 5.0]),
                'very_slow_requests_count': len(df[df['total_time'] > 10.0])
            }
        
        # Error analysis
        if 'success' in df.columns:
            failed_requests = df[df['success'] == False]
            if not failed_requests.empty:
                analysis['error_analysis'] = {
                    'total_failures': len(failed_requests),
                    'failure_rate': len(failed_requests) / len(df),
                    'common_error_stages': failed_requests['pipeline_stage'].value_counts().to_dict() if 'pipeline_stage' in failed_requests.columns else {}
                }
        
        return analysis
    
    def _analyze_with_basic_python(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance patterns using basic Python (fallback when pandas not available)."""
        if not data:
            return {}
        
        total_requests = len(data)
        successful_requests = sum(1 for item in data if item.get('success', False))
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Calculate averages
        total_times = [item.get('total_time', 0) for item in data if 'total_time' in item]
        retrieval_times = [item.get('retrieval_time', 0) for item in data if 'retrieval_time' in item]
        generation_times = [item.get('generation_time', 0) for item in data if 'generation_time' in item]
        documents_retrieved = [item.get('documents_retrieved', 0) for item in data if 'documents_retrieved' in item]
        
        analysis = {
            'summary_stats': {
                'total_requests': total_requests,
                'success_rate': success_rate,
                'avg_total_time': sum(total_times) / len(total_times) if total_times else 0,
                'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
                'avg_generation_time': sum(generation_times) / len(generation_times) if generation_times else 0,
                'avg_documents_retrieved': sum(documents_retrieved) / len(documents_retrieved) if documents_retrieved else 0
            }
        }
        
        # Basic time-based analysis
        hourly_requests = defaultdict(int)
        daily_requests = defaultdict(int)
        
        for item in data:
            if 'timestamp' in item:
                try:
                    timestamp = datetime.fromisoformat(item['timestamp'])
                    hourly_requests[timestamp.hour] += 1
                    daily_requests[timestamp.strftime('%A')] += 1
                except ValueError:
                    continue
        
        analysis['hourly_patterns'] = {
            'requests_by_hour': dict(hourly_requests),
            'avg_response_time_by_hour': {}  # Would need more complex calculation
        }
        
        analysis['daily_patterns'] = {
            'requests_by_day': dict(daily_requests),
            'avg_response_time_by_day': {}  # Would need more complex calculation
        }
        
        # Performance distribution analysis
        if total_times:
            total_times_sorted = sorted(total_times)
            n = len(total_times_sorted)
            
            analysis['performance_distribution'] = {
                'percentiles': {
                    '50th': total_times_sorted[int(n * 0.5)] if n > 0 else 0,
                    '90th': total_times_sorted[int(n * 0.9)] if n > 0 else 0,
                    '95th': total_times_sorted[int(n * 0.95)] if n > 0 else 0,
                    '99th': total_times_sorted[int(n * 0.99)] if n > 0 else 0
                },
                'slow_requests_count': sum(1 for t in total_times if t > 5.0),
                'very_slow_requests_count': sum(1 for t in total_times if t > 10.0)
            }
        
        # Error analysis
        failed_requests = [item for item in data if not item.get('success', False)]
        if failed_requests:
            error_stages = Counter(item.get('pipeline_stage', 'unknown') for item in failed_requests)
            analysis['error_analysis'] = {
                'total_failures': len(failed_requests),
                'failure_rate': len(failed_requests) / total_requests,
                'common_error_stages': dict(error_stages.most_common(5))
            }
        
        return analysis
    
    def generate_performance_dashboard(self, data, output_dir: str = 'dashboard_output'):
        """Generate performance visualization dashboard."""
        if not HAS_PLOTTING:
            print("Dashboard generation requires matplotlib and seaborn. Please install them.")
            return
        
        if (HAS_PANDAS and isinstance(data, pd.DataFrame) and data.empty) or \
           (not HAS_PANDAS and isinstance(data, list) and not data):
            self.analyzer_logger.warning("No data available for dashboard generation")
            return
        
        # Convert to DataFrame if needed
        if not HAS_PANDAS or not isinstance(data, pd.DataFrame):
            if not data:
                return
            df = pd.DataFrame(data) if HAS_PANDAS else None
            if df is None:
                print("Dashboard generation requires pandas. Please install it.")
                return
        else:
            df = data
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Set style
            plt.style.use('default')  # Use default instead of seaborn-v0_8 for compatibility
            if HAS_PLOTTING:
                sns.set_palette("husl")
            
            # 1. Response Time Distribution
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('RAG Pipeline Performance Dashboard', fontsize=16, fontweight='bold')
            
            # Response time histogram
            if 'total_time' in df.columns:
                axes[0, 0].hist(df['total_time'], bins=30, alpha=0.7, edgecolor='black')
                mean_time = df['total_time'].mean()
                axes[0, 0].axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}s')
                axes[0, 0].set_xlabel('Total Response Time (seconds)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Response Time Distribution')
                axes[0, 0].legend()
            
            # Success rate over time
            if 'timestamp' in df.columns and 'success' in df.columns:
                daily_success = df.groupby(df['timestamp'].dt.date)['success'].mean()
                axes[0, 1].plot(daily_success.index, daily_success.values, marker='o')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Success Rate')
                axes[0, 1].set_title('Success Rate Over Time')
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # Response time components
            time_columns = ['retrieval_time', 'generation_time']
            available_time_columns = [col for col in time_columns if col in df.columns]
            if available_time_columns:
                time_data = df[available_time_columns].mean()
                axes[1, 0].bar(time_data.index, time_data.values)
                axes[1, 0].set_xlabel('Pipeline Stage')
                axes[1, 0].set_ylabel('Average Time (seconds)')
                axes[1, 0].set_title('Average Time by Pipeline Stage')
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # Documents retrieved distribution
            if 'documents_retrieved' in df.columns:
                doc_counts = df['documents_retrieved'].value_counts().sort_index()
                axes[1, 1].bar(doc_counts.index, doc_counts.values)
                axes[1, 1].set_xlabel('Number of Documents Retrieved')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Documents Retrieved Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Hourly patterns
            if 'timestamp' in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Requests by hour
                hourly_requests = df.groupby(df['timestamp'].dt.hour).size()
                axes[0].bar(hourly_requests.index, hourly_requests.values)
                axes[0].set_xlabel('Hour of Day')
                axes[0].set_ylabel('Number of Requests')
                axes[0].set_title('Request Volume by Hour')
                axes[0].set_xticks(range(0, 24, 2))
                
                # Response time by hour
                if 'total_time' in df.columns:
                    hourly_time = df.groupby(df['timestamp'].dt.hour)['total_time'].mean()
                    axes[1].plot(hourly_time.index, hourly_time.values, marker='o', linewidth=2)
                    axes[1].set_xlabel('Hour of Day')
                    axes[1].set_ylabel('Average Response Time (seconds)')
                    axes[1].set_title('Response Time by Hour')
                    axes[1].set_xticks(range(0, 24, 2))
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'hourly_patterns.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            self.analyzer_logger.info(f"Dashboard generated in {output_dir}")
            
        except Exception as e:
            print(f"Error generating dashboard: {e}")
            print("This might be due to missing data columns or plotting library issues.")
    
    def identify_performance_bottlenecks(self, data) -> Dict[str, Any]:
        """Identify performance bottlenecks in the RAG pipeline."""
        if (HAS_PANDAS and isinstance(data, pd.DataFrame) and data.empty) or \
           (not HAS_PANDAS and isinstance(data, list) and not data):
            return {}
        
        bottlenecks = {
            'slow_queries': [],
            'high_error_stages': [],
            'inefficient_retrievals': [],
            'recommendations': []
        }
        
        # Handle both pandas DataFrame and list
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            return self._identify_bottlenecks_pandas(data, bottlenecks)
        else:
            return self._identify_bottlenecks_basic(data, bottlenecks)
    
    def _identify_bottlenecks_pandas(self, df, bottlenecks):
        """Identify bottlenecks using pandas."""
        # Identify slow queries
        if 'total_time' in df.columns:
            slow_threshold = df['total_time'].quantile(0.95)
            slow_queries = df[df['total_time'] > slow_threshold]
            
            for _, query in slow_queries.iterrows():
                bottlenecks['slow_queries'].append({
                    'request_id': query.get('request_id', 'unknown'),
                    'query': str(query.get('query', ''))[:100],
                    'total_time': query.get('total_time', 0),
                    'retrieval_time': query.get('retrieval_time', 0),
                    'generation_time': query.get('generation_time', 0)
                })
        
        # Identify high error stages
        if 'success' in df.columns and 'pipeline_stage' in df.columns:
            failed_requests = df[df['success'] == False]
            if not failed_requests.empty:
                error_stages = failed_requests['pipeline_stage'].value_counts()
                bottlenecks['high_error_stages'] = error_stages.head(5).to_dict()
        
        # Identify inefficient retrievals
        if 'retrieval_time' in df.columns and 'documents_retrieved' in df.columns:
            # Avoid division by zero
            df_with_time = df[df['retrieval_time'] > 0]
            if not df_with_time.empty:
                df_with_time = df_with_time.copy()
                df_with_time['retrieval_efficiency'] = df_with_time['documents_retrieved'] / df_with_time['retrieval_time']
                inefficient = df_with_time[df_with_time['retrieval_efficiency'] < df_with_time['retrieval_efficiency'].quantile(0.1)]
                
                for _, retrieval in inefficient.iterrows():
                    bottlenecks['inefficient_retrievals'].append({
                        'request_id': retrieval.get('request_id', 'unknown'),
                        'retrieval_time': retrieval.get('retrieval_time', 0),
                        'documents_retrieved': retrieval.get('documents_retrieved', 0),
                        'efficiency_score': retrieval.get('retrieval_efficiency', 0)
                    })
        
        return self._generate_recommendations(df, bottlenecks)
    
    def _identify_bottlenecks_basic(self, data, bottlenecks):
        """Identify bottlenecks using basic Python."""
        total_times = [item.get('total_time', 0) for item in data if 'total_time' in item and item['total_time'] > 0]
        
        if total_times:
            total_times_sorted = sorted(total_times)
            slow_threshold = total_times_sorted[int(len(total_times_sorted) * 0.95)] if total_times_sorted else 0
            
            for item in data:
                if item.get('total_time', 0) > slow_threshold:
                    bottlenecks['slow_queries'].append({
                        'request_id': item.get('request_id', 'unknown'),
                        'query': str(item.get('query', ''))[:100],
                        'total_time': item.get('total_time', 0),
                        'retrieval_time': item.get('retrieval_time', 0),
                        'generation_time': item.get('generation_time', 0)
                    })
        
        # Identify high error stages
        failed_requests = [item for item in data if not item.get('success', False)]
        if failed_requests:
            error_stages = Counter(item.get('pipeline_stage', 'unknown') for item in failed_requests)
            bottlenecks['high_error_stages'] = dict(error_stages.most_common(5))
        
        return self._generate_recommendations_basic(data, bottlenecks)
    
    def _generate_recommendations(self, df, bottlenecks):
        """Generate recommendations using pandas."""
        recommendations = []
        
        if bottlenecks['slow_queries']:
            recommendations.append(f"Found {len(bottlenecks['slow_queries'])} slow queries. Consider optimizing query processing or increasing timeout thresholds.")
        
        if bottlenecks['high_error_stages']:
            top_error_stage = max(bottlenecks['high_error_stages'].items(), key=lambda x: x[1])
            recommendations.append(f"Stage '{top_error_stage[0]}' has the highest error rate ({top_error_stage[1]} failures). Focus debugging efforts here.")
        
        if bottlenecks['inefficient_retrievals']:
            recommendations.append(f"Found {len(bottlenecks['inefficient_retrievals'])} inefficient retrievals. Consider optimizing vector search or indexing strategy.")
        
        if 'total_time' in df.columns:
            avg_retrieval = df['retrieval_time'].mean() if 'retrieval_time' in df.columns else 0
            avg_generation = df['generation_time'].mean() if 'generation_time' in df.columns else 0
            
            if avg_retrieval > 0 and avg_generation > 0:
                if avg_retrieval > avg_generation * 2:
                    recommendations.append("Retrieval time is significantly higher than generation time. Consider optimizing vector search performance.")
                elif avg_generation > avg_retrieval * 3:
                    recommendations.append("Generation time is significantly higher than retrieval time. Consider using a faster language model or optimizing prompts.")
        
        bottlenecks['recommendations'] = recommendations
        return bottlenecks
    
    def _generate_recommendations_basic(self, data, bottlenecks):
        """Generate recommendations using basic Python."""
        recommendations = []
        
        if bottlenecks['slow_queries']:
            recommendations.append(f"Found {len(bottlenecks['slow_queries'])} slow queries. Consider optimizing query processing or increasing timeout thresholds.")
        
        if bottlenecks['high_error_stages']:
            top_error = max(bottlenecks['high_error_stages'].items(), key=lambda x: x[1]) if bottlenecks['high_error_stages'] else None
            if top_error:
                recommendations.append(f"Stage '{top_error[0]}' has the highest error rate ({top_error[1]} failures). Focus debugging efforts here.")
        
        if bottlenecks['inefficient_retrievals']:
            recommendations.append(f"Found {len(bottlenecks['inefficient_retrievals'])} inefficient retrievals. Consider optimizing vector search or indexing strategy.")
        
        # Basic time comparison
        retrieval_times = [item.get('retrieval_time', 0) for item in data if 'retrieval_time' in item]
        generation_times = [item.get('generation_time', 0) for item in data if 'generation_time' in item]
        
        if retrieval_times and generation_times:
            avg_retrieval = sum(retrieval_times) / len(retrieval_times)
            avg_generation = sum(generation_times) / len(generation_times)
            
            if avg_retrieval > avg_generation * 2:
                recommendations.append("Retrieval time is significantly higher than generation time. Consider optimizing vector search performance.")
            elif avg_generation > avg_retrieval * 3:
                recommendations.append("Generation time is significantly higher than retrieval time. Consider using a faster language model or optimizing prompts.")
        
        bottlenecks['recommendations'] = recommendations
        return bottlenecks
    
    def export_analysis_report(self, output_path: str = 'rag_analysis_report.json', days_back: int = 7):
        """Export comprehensive analysis report."""
        self.analyzer_logger.info(f"Generating comprehensive analysis report")
        
        # Load data
        data = self.load_all_metrics(days_back)
        
        if (HAS_PANDAS and isinstance(data, pd.DataFrame) and data.empty) or \
           (not HAS_PANDAS and isinstance(data, list) and not data):
            report = {
                'generated_at': datetime.now().isoformat(),
                'data_period_days': days_back,
                'status': 'No data available for analysis'
            }
        else:
            # Perform all analyses
            performance_patterns = self.analyze_performance_patterns(data)
            bottlenecks = self.identify_performance_bottlenecks(data)
            
            # Get data summary
            if HAS_PANDAS and isinstance(data, pd.DataFrame):
                data_summary = {
                    'total_records': len(data),
                    'date_range': {
                        'start': data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                        'end': data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
                    }
                }
                data_quality = {
                    'missing_timestamps': data['timestamp'].isna().sum() if 'timestamp' in data.columns else 0,
                    'missing_success_flags': data['success'].isna().sum() if 'success' in data.columns else 0,
                    'missing_total_times': data['total_time'].isna().sum() if 'total_time' in data.columns else 0
                }
            else:
                data_summary = {
                    'total_records': len(data),
                    'date_range': {
                        'start': min(item.get('timestamp', '') for item in data if 'timestamp' in item) or None,
                        'end': max(item.get('timestamp', '') for item in data if 'timestamp' in item) or None
                    }
                }
                data_quality = {
                    'missing_timestamps': sum(1 for item in data if 'timestamp' not in item),
                    'missing_success_flags': sum(1 for item in data if 'success' not in item),
                    'missing_total_times': sum(1 for item in data if 'total_time' not in item)
                }
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'data_period_days': days_back,
                'data_summary': data_summary,
                'performance_patterns': performance_patterns,
                'bottlenecks': bottlenecks,
                'data_quality': data_quality
            }
        
        # Export report
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.analyzer_logger.info(f"Analysis report exported to {output_path}")
        except Exception as e:
            print(f"Error exporting report: {e}")
        
        return report


class RAGLogMonitor:
    """Real-time monitoring for RAG pipeline logs."""
    
    def __init__(self, logger: RAGPipelineLogger, check_interval: int = 60):
        self.logger = logger
        self.check_interval = check_interval
        self.monitor_logger = logger.get_logger('monitoring')
        self.last_check = datetime.now()
        self.alert_thresholds = {
            'success_rate_min': 0.95,
            'avg_response_time_max': 5.0,
            'error_rate_max': 0.05,
            'queue_length_max': 100
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check current system health metrics."""
        self.monitor_logger.info("Performing system health check")
        
        # Get recent metrics (last hour)
        recent_logs = self.logger.get_temp_logs_for_analysis(log_type='metrics')
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        recent_metrics = []
        for log in recent_logs:
            try:
                log_time = datetime.fromisoformat(log['data']['timestamp'])
                if log_time > cutoff_time:
                    recent_metrics.append(log['data'])
            except (KeyError, ValueError):
                continue
        
        if not recent_metrics:
            return {
                'status': 'NO_DATA',
                'message': 'No recent metrics available',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate health metrics
        total_requests = len(recent_metrics)
        successful_requests = sum(1 for m in recent_metrics if m.get('success', False))
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        response_times = [m.get('total_time', 0) for m in recent_metrics]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        error_rate = 1 - success_rate
        
        # Determine overall health status
        alerts = []
        
        if success_rate < self.alert_thresholds['success_rate_min']:
            alerts.append(f"Low success rate: {success_rate:.2%} (threshold: {self.alert_thresholds['success_rate_min']:.2%})")
        
        if avg_response_time > self.alert_thresholds['avg_response_time_max']:
            alerts.append(f"High response time: {avg_response_time:.2f}s (threshold: {self.alert_thresholds['avg_response_time_max']}s)")
        
        if error_rate > self.alert_thresholds['error_rate_max']:
            alerts.append(f"High error rate: {error_rate:.2%} (threshold: {self.alert_thresholds['error_rate_max']:.2%})")
        
        # Determine status
        if not alerts:
            status = 'HEALTHY'
        elif len(alerts) == 1:
            status = 'WARNING'
        else:
            status = 'CRITICAL'
        
        health_report = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_requests_last_hour': total_requests,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'error_rate': error_rate
            },
            'alerts': alerts,
            'thresholds': self.alert_thresholds
        }
        
        # Log health status
        if status == 'HEALTHY':
            self.monitor_logger.info(f"System health: {status}")
        elif status == 'WARNING':
            self.monitor_logger.warning(f"System health: {status} - {'; '.join(alerts)}")
        else:
            self.monitor_logger.error(f"System health: {status} - {'; '.join(alerts)}")
        
        return health_report
    
    def start_monitoring(self, duration_minutes: Optional[int] = None):
        """Start continuous monitoring."""
        self.monitor_logger.info(f"Starting continuous monitoring (check interval: {self.check_interval}s)")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes) if duration_minutes else None
        
        try:
            while True:
                # Check if we should stop
                if end_time and datetime.now() > end_time:
                    break
                
                # Perform health check
                health_report = self.check_system_health()
                
                # Save health report
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                health_file = f"health_check_{timestamp}.json"
                
                # Create monitoring directory if it doesn't exist
                monitoring_dir = self.logger._get_temp_log_path('monitoring', '')
                os.makedirs(monitoring_dir, exist_ok=True)
                
                health_path = self.logger._get_temp_log_path('monitoring', health_file)
                
                try:
                    with open(health_path, 'w') as f:
                        json.dump(health_report, f, indent=2, default=str)
                except Exception as e:
                    print(f"Warning: Could not save health report: {e}")
                
                # Print status to console
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Health Status: {health_report['status']}")
                if health_report['alerts']:
                    for alert in health_report['alerts']:
                        print(f"  ⚠️  {alert}")
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.monitor_logger.info("Monitoring stopped by user")
        except Exception as e:
            self.monitor_logger.error(f"Monitoring error: {e}")
        
        self.monitor_logger.info("Monitoring stopped")


# Command-line interface for log analysis
def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze RAG pipeline logs')
    parser.add_argument('--config', default='rag_logging_config.yaml', help='Logger configuration file')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--output-dir', default='analysis_output', help='Output directory for reports')
    parser.add_argument('--generate-dashboard', action='store_true', help='Generate visualization dashboard')
    parser.add_argument('--export-report', action='store_true', help='Export comprehensive analysis report')
    parser.add_argument('--monitor', action='store_true', help='Start real-time monitoring')
    parser.add_argument('--monitor-duration', type=int, default=60, help='Monitor duration in minutes')
    
    args = parser.parse_args()
    
    # Initialize logger and analyzer
    try:
        logger = RAGPipelineLogger(args.config)
        print(f"✅ Logger initialized with config: {args.config}")
    except Exception as e:
        print(f"⚠️ Error loading config {args.config}: {e}")
        print("🔄 Using default configuration...")
        logger = RAGPipelineLogger()
    
    analyzer = RAGLogAnalyzer(logger)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.monitor:
        # Start monitoring mode
        monitor = RAGLogMonitor(logger)
        print(f"🔍 Starting system health monitoring for {args.monitor_duration} minutes...")
        print("Press Ctrl+C to stop monitoring early")
        monitor.start_monitoring(args.monitor_duration)
        return
    
    # Load data
    print(f"📊 Loading metrics data for last {args.days} days...")
    data = analyzer.load_all_metrics(args.days)
    
    if (HAS_PANDAS and isinstance(data, pd.DataFrame) and data.empty) or \
       (not HAS_PANDAS and isinstance(data, list) and not data):
        print("❌ No metrics data found for analysis.")
        print("💡 Make sure your RAG pipeline is running and generating logs.")
        return
    
    data_count = len(data)
    print(f"✅ Loaded {data_count} metrics records")
    
    # Generate dashboard
    if args.generate_dashboard:
        print("📈 Generating performance dashboard...")
        analyzer.generate_performance_dashboard(data, args.output_dir)
        print(f"✅ Dashboard saved to {args.output_dir}")
    
    # Export comprehensive report
    if args.export_report:
        print("📄 Generating comprehensive analysis report...")
        report_path = os.path.join(args.output_dir, 'comprehensive_analysis_report.json')
        report = analyzer.export_analysis_report(report_path, args.days)
        print(f"✅ Report saved to {report_path}")
        
        # Print summary
        if 'performance_patterns' in report:
            patterns = report['performance_patterns']
            if 'summary_stats' in patterns:
                stats = patterns['summary_stats']
                print(f"\n=== 📊 Performance Summary ===")
                print(f"Total Requests: {stats.get('total_requests', 0)}")
                print(f"Success Rate: {stats.get('success_rate', 0):.2%}")
                print(f"Average Response Time: {stats.get('avg_total_time', 0):.2f}s")
                print(f"Average Documents Retrieved: {stats.get('avg_documents_retrieved', 0):.1f}")
        
        if 'bottlenecks' in report:
            bottlenecks = report['bottlenecks']
            if bottlenecks.get('recommendations'):
                print(f"\n=== 💡 Recommendations ===")
                for i, rec in enumerate(bottlenecks['recommendations'], 1):
                    print(f"{i}. {rec}")
    
    # Always show basic analysis
    print(f"\n=== 📋 Basic Analysis ===")
    analysis = analyzer.analyze_performance_patterns(data)
    
    if 'summary_stats' in analysis:
        stats = analysis['summary_stats']
        print(f"📊 Total Requests: {stats.get('total_requests', 0)}")
        print(f"✅ Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"⏱️ Average Response Time: {stats.get('avg_total_time', 0):.2f}s")
        print(f"🔍 Average Retrieval Time: {stats.get('avg_retrieval_time', 0):.2f}s")
        print(f"🤖 Average Generation Time: {stats.get('avg_generation_time', 0):.2f}s")
    
    # Show performance distribution
    if 'performance_distribution' in analysis:
        perf = analysis['performance_distribution']
        print(f"\n=== ⚡ Response Time Percentiles ===")
        for percentile, value in perf.get('percentiles', {}).items():
            print(f"{percentile}: {value:.2f}s")
        
        slow_count = perf.get('slow_requests_count', 0)
        very_slow_count = perf.get('very_slow_requests_count', 0)
        total_requests = stats.get('total_requests', 1)
        
        if slow_count > 0:
            print(f"\n🐌 Slow Requests (>5s): {slow_count} ({slow_count/total_requests:.1%})")
        if very_slow_count > 0:
            print(f"🐌 Very Slow Requests (>10s): {very_slow_count} ({very_slow_count/total_requests:.1%})")
    
    # Show bottlenecks if any
    bottlenecks = analyzer.identify_performance_bottlenecks(data)
    if bottlenecks.get('recommendations'):
        print(f"\n=== 🔧 Performance Recommendations ===")
        for i, rec in enumerate(bottlenecks['recommendations'], 1):
            print(f"{i}. {rec}")


# Utility functions for quick analysis
def quick_performance_summary(config_path: str = 'rag_logging_config.yaml', days: int = 1):
    """Generate a quick performance summary."""
    try:
        logger = RAGPipelineLogger(config_path)
    except:
        logger = RAGPipelineLogger()
    
    analyzer = RAGLogAnalyzer(logger)
    
    data = analyzer.load_all_metrics(days)
    if (HAS_PANDAS and isinstance(data, pd.DataFrame) and data.empty) or \
       (not HAS_PANDAS and isinstance(data, list) and not data):
        print("No data available for analysis")
        return
    
    analysis = analyzer.analyze_performance_patterns(data)
    
    print(f"=== RAG Pipeline Performance Summary (Last {days} day{'s' if days != 1 else ''}) ===")
    
    if 'summary_stats' in analysis:
        stats = analysis['summary_stats']
        print(f"📊 Total Requests: {stats.get('total_requests', 0)}")
        print(f"✅ Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"⏱️ Average Response Time: {stats.get('avg_total_time', 0):.2f}s")
        print(f"🔍 Average Documents Retrieved: {stats.get('avg_documents_retrieved', 0):.1f}")
        print(f"📥 Average Retrieval Time: {stats.get('avg_retrieval_time', 0):.2f}s")
        print(f"🤖 Average Generation Time: {stats.get('avg_generation_time', 0):.2f}s")


def monitor_system_health(config_path: str = 'rag_logging_config.yaml', duration_minutes: int = 60):
    """Start system health monitoring."""
    try:
        logger = RAGPipelineLogger(config_path)
    except:
        logger = RAGPipelineLogger()
    
    monitor = RAGLogMonitor(logger)
    
    print(f"🔍 Starting system health monitoring for {duration_minutes} minutes...")
    print("Press Ctrl+C to stop monitoring early")
    
    monitor.start_monitoring(duration_minutes)


def create_sample_logs():
    """Create sample logs for testing the analyzer."""
    print("📝 Creating sample logs for testing...")
    
    # Create directories
    os.makedirs('temp_logs/metrics', exist_ok=True)
    
    # Generate sample metrics data
    sample_data = []
    base_time = datetime.now() - timedelta(days=1)
    
    for i in range(50):  # Create 50 sample requests
        timestamp = base_time + timedelta(minutes=i*10)
        
        sample_request = {
            'request_id': f'req_{i:03d}',
            'timestamp': timestamp.isoformat(),
            'query': f'Sample query {i}',
            'total_time': 0.5 + (i % 10) * 0.2,  # Vary between 0.5 and 2.3 seconds
            'retrieval_time': 0.1 + (i % 5) * 0.05,  # Vary between 0.1 and 0.3 seconds
            'generation_time': 0.3 + (i % 8) * 0.1,  # Vary between 0.3 and 1.0 seconds
            'documents_retrieved': 3 + (i % 5),  # Vary between 3 and 7 documents
            'similarity_scores': [0.9 - (j * 0.05) for j in range(3 + (i % 5))],
            'tokens_used': 100 + (i % 20) * 10,  # Vary between 100 and 290 tokens
            'response_length': 200 + (i % 30) * 20,  # Vary response length
            'success': i % 20 != 0,  # 95% success rate (fail every 20th request)
            'error_message': 'Timeout error' if i % 20 == 0 else '',
            'pipeline_stage': 'text_generation' if i % 20 == 0 else 'completed'
        }
        
        sample_data.append(sample_request)
        
        # Save individual metric file
        filename = f'metrics_req_{i:03d}_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        filepath = os.path.join('temp_logs/metrics', filename)
        
        with open(filepath, 'w') as f:
            json.dump(sample_request, f, indent=2, default=str)
    
    print(f"✅ Created {len(sample_data)} sample log files in temp_logs/metrics/")
    return sample_data


def test_analyzer():
    """Test the analyzer with sample data."""
    print("🧪 Testing RAG Log Analyzer...")
    
    # Create sample logs
    create_sample_logs()
    
    # Initialize analyzer
    logger = RAGPipelineLogger()
    analyzer = RAGLogAnalyzer(logger)
    
    # Test loading data
    print("\n📊 Testing data loading...")
    data = analyzer.load_all_metrics(days=2)
    print(f"✅ Loaded {len(data) if isinstance(data, list) else len(data) if HAS_PANDAS else 0} records")
    
    # Test analysis
    print("\n📈 Testing performance analysis...")
    analysis = analyzer.analyze_performance_patterns(data)
    
    if 'summary_stats' in analysis:
        stats = analysis['summary_stats']
        print(f"✅ Analysis complete:")
        print(f"   Total Requests: {stats.get('total_requests', 0)}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"   Avg Response Time: {stats.get('avg_total_time', 0):.2f}s")
    
    # Test bottleneck identification
    print("\n🔍 Testing bottleneck identification...")
    bottlenecks = analyzer.identify_performance_bottlenecks(data)
    
    if bottlenecks.get('recommendations'):
        print(f"✅ Found {len(bottlenecks['recommendations'])} recommendations")
        for i, rec in enumerate(bottlenecks['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
    
    # Test dashboard generation (if plotting available)
    if HAS_PLOTTING:
        print("\n📊 Testing dashboard generation...")
        try:
            analyzer.generate_performance_dashboard(data, 'test_dashboard')
            print("✅ Dashboard generated successfully")
        except Exception as e:
            print(f"⚠️ Dashboard generation failed: {e}")
    else:
        print("\n⚠️ Skipping dashboard test (matplotlib not available)")
    
    # Test report export
    print("\n📄 Testing report export...")
    try:
        report = analyzer.export_analysis_report('test_report.json', days=2)
        print("✅ Report exported successfully")
        print(f"   Report status: {report.get('status', 'Generated')}")
    except Exception as e:
        print(f"⚠️ Report export failed: {e}")
    
    print("\n🎉 Analyzer testing complete!")


if __name__ == "__main__":
    # Check if we're running in test mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_analyzer()
    else:
        main()