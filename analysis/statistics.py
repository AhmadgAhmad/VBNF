# antifragile_vbnf/analysis/statistics.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any

class StatisticalAnalyzer:
    """Statistical analysis for antifragile experiments."""
    
    def __init__(self):
        pass
    
    def compute_antifragile_stats(self, std_vals: List[float], anti_vals: List[float], 
                                 metric_name: str) -> Dict[str, Any]:
        """Compute statistics with antifragile interpretation."""
        std_vals = np.array(std_vals)
        anti_vals = np.array(anti_vals)
        
        # Basic stats
        std_mean = np.mean(std_vals)
        anti_mean = np.mean(anti_vals)
        improvement = ((anti_mean - std_mean) / abs(std_mean)) * 100 if std_mean != 0 else 0
        
        # Statistical test
        t_stat, p_value = stats.ttest_rel(anti_vals, std_vals)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(std_vals)**2 + np.std(anti_vals)**2) / 2)
        cohens_d = (anti_mean - std_mean) / pooled_std if pooled_std != 0 else 0
        
        return {
            'metric': metric_name,
            'std_mean': std_mean,
            'anti_mean': anti_mean,
            'improvement_pct': improvement,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': p_value < 0.05,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
            'antifragile_interpretation': self.interpret_antifragile_result(improvement, p_value, metric_name)
        }
    
    def interpret_antifragile_result(self, improvement: float, p_value: float, metric_name: str) -> str:
        """Provide antifragile-specific interpretation of results."""
        is_significant = p_value < 0.05
        
        antifragile_metrics = ['jensen_gap', 'stress_benefit', 'recovery_superiority', 
                              'recovery_speed', 'convexity', 'single_outlier']
        
        if metric_name in antifragile_metrics:
            if is_significant and improvement > 0:
                return "✅ Antifragile benefit confirmed"
            elif is_significant and improvement < 0:
                return "❌ Unexpected deterioration in antifragile metric"
            else:
                return "➖ No significant antifragile benefit"
        else:
            if is_significant and improvement < 0:
                return "⚠️ Acceptable trade-off (if moderate)"
            elif is_significant and improvement > 0:
                return "🎁 Unexpected bonus improvement"
            else:
                return "➖ No significant change"
    
    def extract_metric_values(self, results_dict: Dict) -> Dict[str, Dict[str, List[float]]]:
        """Extract standard and antifragile metrics from the nested results structure."""
        extracted_metrics = {}
        
        # Process distribution_shift results
        if 'distribution_shift' in results_dict:
            dist_shift = results_dict['distribution_shift']
            # Extract horizontal_shift
            if 'horizontal_shift' in dist_shift:
                horizontal_shift_data = {'std': [], 'anti': []}
                for item in dist_shift['horizontal_shift']:
                    if 'std_log_prob' in item and 'anti_log_prob' in item:
                        horizontal_shift_data['std'].append(float(item['std_log_prob']))
                        horizontal_shift_data['anti'].append(float(item['anti_log_prob']))
                extracted_metrics['horizontal_shift'] = horizontal_shift_data
                
            # Extract noise_variation
            if 'noise_variation' in dist_shift:
                noise_data = {'std': [], 'anti': []}
                for item in dist_shift['noise_variation']:
                    if 'std_log_prob' in item and 'anti_log_prob' in item:
                        noise_data['std'].append(float(item['std_log_prob']))
                        noise_data['anti'].append(float(item['anti_log_prob']))
                extracted_metrics['noise_variation'] = noise_data
                
            # Extract shape_deformation
            if 'shape_deformation' in dist_shift:
                shape_data = {'std': [], 'anti': []}
                for item in dist_shift['shape_deformation']:
                    if 'std_log_prob' in item and 'anti_log_prob' in item:
                        shape_data['std'].append(float(item['std_log_prob']))
                        shape_data['anti'].append(float(item['anti_log_prob']))
                extracted_metrics['shape_deformation'] = shape_data
        
        # Process black_swan results
        if 'black_swan' in results_dict:
            black_swan = results_dict['black_swan']
            # Extract single_outlier
            if 'single_outlier' in black_swan:
                outlier_data = {'std': [], 'anti': []}
                for item in black_swan['single_outlier']:
                    if 'std_impact_pct' in item and 'anti_impact_pct' in item:
                        outlier_data['std'].append(float(item['std_impact_pct']))
                        outlier_data['anti'].append(float(item['anti_impact_pct']))
                extracted_metrics['single_outlier'] = outlier_data
                
            # Extract outlier_cluster
            if 'outlier_cluster' in black_swan:
                cluster_data = {'std': [], 'anti': []}
                for item in black_swan['outlier_cluster']:
                    if 'std_impact_pct' in item and 'anti_impact_pct' in item:
                        cluster_data['std'].append(float(item['std_impact_pct']))
                        cluster_data['anti'].append(float(item['anti_impact_pct']))
                extracted_metrics['outlier_cluster'] = cluster_data
        
        # Process dynamic_environment results
        if 'dynamic_environment' in results_dict:
            dynamic_env = results_dict['dynamic_environment']
            # Extract distribution_drift
            if 'distribution_drift' in dynamic_env:
                drift_data = {'std': [], 'anti': []}
                for item in dynamic_env['distribution_drift']:
                    if 'std_log_prob' in item and 'anti_log_prob' in item:
                        drift_data['std'].append(float(item['std_log_prob']))
                        drift_data['anti'].append(float(item['anti_log_prob']))
                extracted_metrics['distribution_drift'] = drift_data
                
            # Extract oscillation
            if 'oscillation' in dynamic_env:
                oscillation_data = {'std': [], 'anti': []}
                for item in dynamic_env['oscillation']:
                    if 'std_log_prob' in item and 'anti_log_prob' in item:
                        oscillation_data['std'].append(float(item['std_log_prob']))
                        oscillation_data['anti'].append(float(item['anti_log_prob']))
                extracted_metrics['oscillation'] = oscillation_data
        
        # Process transfer_function results
        if 'transfer_function' in results_dict:
            transfer_func = results_dict['transfer_function']
            # Extract stress_level_variation
            if 'stress_level_variation' in transfer_func:
                stress_data = {'std': [], 'anti': []}
                for item in transfer_func['stress_level_variation']:
                    if 'std_fragility_ratio' in item and 'anti_fragility_ratio' in item:
                        stress_data['std'].append(float(item['std_fragility_ratio']))
                        stress_data['anti'].append(float(item['anti_fragility_ratio']))
                extracted_metrics['stress_level_variation'] = stress_data
                
            # Extract extreme_context
            if 'extreme_context' in transfer_func:
                context_data = {'std': [], 'anti': []}
                for item in transfer_func['extreme_context']:
                    if 'std_distance' in item and 'anti_distance' in item:
                        context_data['std'].append(float(item['std_distance']))
                        context_data['anti'].append(float(item['anti_distance']))
                extracted_metrics['extreme_context'] = context_data
        
        # Process advanced_antifragility results
        if 'advanced_antifragility' in results_dict:
            advanced = results_dict['advanced_antifragility']
            # Extract volatility_response
            if 'volatility_response' in advanced:
                volatility_data = {'std': [], 'anti': []}
                for item in advanced['volatility_response']:
                    if 'std_jensen_gap' in item and 'anti_jensen_gap' in item:
                        volatility_data['std'].append(float(item['std_jensen_gap']))
                        volatility_data['anti'].append(float(item['anti_jensen_gap']))
                extracted_metrics['volatility_response'] = volatility_data
                
            # Extract convexity
            if 'convexity' in advanced:
                convexity_data = {'std': [], 'anti': []}
                for item in advanced['convexity']:
                    if 'std_convexity' in item and 'anti_convexity' in item:
                        convexity_data['std'].append(float(item['std_convexity']))
                        convexity_data['anti'].append(float(item['anti_convexity']))
                extracted_metrics['convexity'] = convexity_data
                
            # Extract barbell_strategy
            if 'barbell_strategy' in advanced:
                barbell_data = {'std': [], 'anti': []}
                for item in advanced['barbell_strategy']:
                    if 'std_variance' in item and 'anti_variance' in item:
                        barbell_data['std'].append(float(item['std_variance']))
                        barbell_data['anti'].append(float(item['anti_variance']))
                extracted_metrics['barbell_strategy'] = barbell_data
                
            # Extract recovery_speed
            if 'recovery_speed' in advanced:
                recovery_data = {'std': [], 'anti': []}
                for item in advanced['recovery_speed']:
                    if 'std_recovery_pct' in item and 'anti_recovery_pct' in item:
                        recovery_data['std'].append(float(item['std_recovery_pct']))
                        recovery_data['anti'].append(float(item['anti_recovery_pct']))
                extracted_metrics['recovery_speed'] = recovery_data
        
        return extracted_metrics
    
    def analyze_multiple_runs(self, all_run_metrics: Dict[str, Dict[str, List[float]]]) -> List[Dict]:
        """Analyze results across multiple runs."""
        stats_results = []
        
        # Define which metrics are better with higher values
        higher_better_metrics = [
            'volatility_response',  # Jensen's gap - higher is more antifragile
            'convexity',            # Higher convexity is better for antifragility
            'recovery_speed'        # Faster recovery is better
        ]
        
        for metric_name, metric_data in all_run_metrics.items():
            if len(metric_data['std']) == 0 or len(metric_data['anti']) == 0:
                continue
                
            is_higher_better = metric_name in higher_better_metrics
            stat_result = self.compute_stats(
                metric_data['std'], metric_data['anti'], metric_name, is_higher_better
            )
            stats_results.append(stat_result)
        
        return stats_results
    
    def compute_stats(self, std_vals: List[float], anti_vals: List[float], 
                     metric_name: str, is_higher_better: bool = True) -> Dict:
        """Compute statistical significance with direction consideration."""
        if len(std_vals) == 0 or len(anti_vals) == 0:
            return {
                'metric': metric_name,
                'std_mean': float('nan'),
                'std_std': float('nan'),
                'anti_mean': float('nan'),
                'anti_std': float('nan'),
                'improvement': float('nan'),
                'p_value': float('nan'),
                'is_significant': False,
                'better_method': "No data"
            }
        
        # For some metrics, lower values are better
        if not is_higher_better:
            std_vals = [-x for x in std_vals]
            anti_vals = [-x for x in anti_vals]
        
        # Convert to numpy arrays for calculations
        std_vals = np.array(std_vals)
        anti_vals = np.array(anti_vals)
        
        # Compute mean and standard deviation
        std_mean = np.mean(std_vals)
        std_std = np.std(std_vals)
        anti_mean = np.mean(anti_vals)
        anti_std = np.std(anti_vals)
        
        # Compute improvement percentage
        improvement = ((anti_mean - std_mean) / abs(std_mean)) * 100 if std_mean != 0 else float('nan')
        
        # Compute p-value using paired t-test
        t_stat, p_value = stats.ttest_rel(anti_vals, std_vals)
        
        # Determine if result is statistically significant (p < 0.05)
        is_significant = p_value < 0.05
        
        # Determine which method is better
        if anti_mean > std_mean:
            better_method = "A-vbnf"
        else:
            better_method = "Standard vbnf"
        
        return {
            'metric': metric_name,
            'std_mean': std_mean,
            'std_std': std_std,
            'anti_mean': anti_mean,
            'anti_std': anti_std,
            'improvement': improvement,
            'p_value': p_value,
            'is_significant': is_significant,
            'better_method': better_method
        }
    
    def generate_summary_report(self, stats_results: List[Dict]) -> str:
        """Generate a comprehensive summary report."""
        significant_improvements = [r for r in stats_results if r['is_significant'] and r['better_method'] == 'A-vbnf']
        significant_deteriorations = [r for r in stats_results if r['is_significant'] and r['better_method'] == 'Standard vbnf']
        
        report = f"""
ANTIFRAGILE vbnf PERFORMANCE ANALYSIS
=====================================

Total Metrics Tested: {len(stats_results)}
Significant Improvements: {len(significant_improvements)}
Significant Deteriorations: {len(significant_deteriorations)}
No Significant Difference: {len(stats_results) - len(significant_improvements) - len(significant_deteriorations)}

SUCCESS RATE: {len(significant_improvements)/len(stats_results)*100:.1f}%

DETAILED RESULTS:
"""
        
        for result in stats_results:
            status = "✅" if result['is_significant'] and result['better_method'] == 'A-vbnf' else \
                    "❌" if result['is_significant'] and result['better_method'] == 'Standard vbnf' else "➖"
            
            report += f"{status} {result['metric']:<25} | Improvement: {result['improvement']:+6.2f}% | p-value: {result['p_value']:.4f}\n"
        
        return report

# Factory functions for backward compatibility
def extract_metric_values(results_dict: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Factory function for extracting metric values."""
    analyzer = StatisticalAnalyzer()
    return analyzer.extract_metric_values(results_dict)

def compute_antifragile_stats(std_vals: List[float], anti_vals: List[float], 
                             metric_name: str) -> Dict[str, Any]:
    """Factory function for computing antifragile statistics."""
    analyzer = StatisticalAnalyzer()
    return analyzer.compute_antifragile_stats(std_vals, anti_vals, metric_name)