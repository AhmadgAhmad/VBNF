# antifragile_vbnf/analysis/antifragile_analysis.py
import numpy as np
from typing import Dict, List, Any
from .statistics import StatisticalAnalyzer

class AntifragileAnalyzer:
    """Specialized analyzer for antifragile-specific metrics and hypotheses."""
    
    def __init__(self):
        self.stats_analyzer = StatisticalAnalyzer()
    
    def calculate_antifragile_metrics(self, all_results: Dict) -> Dict[str, Dict[str, List[float]]]:
        """Calculate antifragile-specific metrics from existing test results."""
        antifragile_metrics = {}
        
        # 1. Convexity Response (Jensen's Gap)
        antifragile_metrics['jensen_gap'] = self.calculate_jensen_gap(all_results)
        
        # 2. Stress Benefit Ratio
        antifragile_metrics['stress_benefit'] = self.calculate_stress_benefit(all_results)
        
        # 3. Recovery Superiority
        antifragile_metrics['recovery_superiority'] = self.calculate_recovery_superiority(all_results)
        
        # 4. Interpolation Advantage
        antifragile_metrics['interpolation_advantage'] = self.calculate_interpolation_advantage(all_results)
        
        # 5. Tail Event Preparedness
        antifragile_metrics['tail_preparedness'] = self.calculate_tail_preparedness(all_results)
        
        return antifragile_metrics
    
    def calculate_jensen_gap(self, all_results: Dict) -> Dict[str, List[float]]:
        """Calculate Jensen's gap: E[f(stress)] - f(E[stress])."""
        jensen_gap = {'std': [], 'anti': []}
        
        # Use stress_level_variation and noise_variation tests
        stress_results = all_results.get('stress_level_variation', {})
        noise_results = all_results.get('noise_variation', {})
        
        for model_type in ['std', 'anti']:
            if model_type in stress_results:
                stress_perfs = stress_results[model_type]
                if len(stress_perfs) >= 3:  # Need multiple stress levels
                    # Calculate E[f(stress)]
                    expected_performance = np.mean(stress_perfs)
                    
                    # Calculate f(E[stress]) - performance at average stress
                    mid_performance = stress_perfs[len(stress_perfs)//2]
                    
                    # Jensen's gap
                    gap = expected_performance - mid_performance
                    jensen_gap[model_type].append(gap)
        
        return jensen_gap
    
    def calculate_stress_benefit(self, all_results: Dict) -> Dict[str, List[float]]:
        """Measure relative improvement under stress vs baseline."""
        stress_benefit = {'std': [], 'anti': []}
        
        # Compare extreme_context vs normal performance
        extreme_results = all_results.get('extreme_context', {})
        baseline_results = all_results.get('horizontal_shift', {})  # Using as baseline
        
        for model_type in ['std', 'anti']:
            if model_type in extreme_results and model_type in baseline_results:
                extreme_perf = np.mean(extreme_results[model_type])
                baseline_perf = np.mean(baseline_results[model_type])
                
                # Relative improvement under stress
                relative_benefit = (extreme_perf - baseline_perf) / abs(baseline_perf)
                stress_benefit[model_type].append(relative_benefit)
        
        return stress_benefit
    
    def calculate_recovery_superiority(self, all_results: Dict) -> Dict[str, List[float]]:
        """Measure recovery capability from disruptions."""
        recovery_superiority = {'std': [], 'anti': []}
        
        # Use oscillation test as proxy for recovery
        oscillation_results = all_results.get('oscillation', {})
        
        for model_type in ['std', 'anti']:
            if model_type in oscillation_results:
                perf_series = oscillation_results[model_type]
                if len(perf_series) >= 5:
                    # Measure recovery rate (improvement trend in latter half)
                    latter_half = perf_series[len(perf_series)//2:]
                    recovery_rate = np.polyfit(range(len(latter_half)), latter_half, 1)[0]
                    recovery_superiority[model_type].append(recovery_rate)
        
        return recovery_superiority
    
    def calculate_interpolation_advantage(self, all_results: Dict) -> Dict[str, List[float]]:
        """Measure performance in critical interpolation regions."""
        interpolation_advantage = {'std': [], 'anti': []}
        
        # Use distribution_drift as proxy for interpolation challenges
        drift_results = all_results.get('distribution_drift', {})
        
        for model_type in ['std', 'anti']:
            if model_type in drift_results:
                # Focus on middle transition points
                drift_perfs = drift_results[model_type]
                if len(drift_perfs) >= 5:
                    middle_region = drift_perfs[len(drift_perfs)//3:2*len(drift_perfs)//3]
                    interpolation_quality = np.mean(middle_region)
                    interpolation_advantage[model_type].append(interpolation_quality)
        
        return interpolation_advantage
    
    def calculate_tail_preparedness(self, all_results: Dict) -> Dict[str, List[float]]:
        """Measure handling of extreme/tail events."""
        tail_preparedness = {'std': [], 'anti': []}
        
        # Combine single_outlier and outlier_cluster results
        single_results = all_results.get('single_outlier', {})
        cluster_results = all_results.get('outlier_cluster', {})
        
        for model_type in ['std', 'anti']:
            tail_scores = []
            
            if model_type in single_results:
                tail_scores.extend(single_results[model_type])
            if model_type in cluster_results:
                tail_scores.extend(cluster_results[model_type])
            
            if tail_scores:
                tail_preparedness[model_type].append(np.mean(tail_scores))
        
        return tail_preparedness
    
    def analyze_antifragile_results(self, all_run_metrics: Dict[str, Dict[str, List[float]]]) -> Dict:
        """Analyze results from antifragile perspective."""
        analysis = {
            'antifragile_hypothesis_tests': {},
            'trade_off_analysis': {},
            'conditional_performance': {}
        }
        
        # Define antifragile-focused metrics (where improvement is expected)
        antifragile_core_metrics = [
            'jensen_gap', 'stress_benefit', 'recovery_superiority', 
            'interpolation_advantage', 'tail_preparedness',
            'recovery_speed', 'convexity', 'single_outlier'
        ]
        
        # Define acceptable trade-off metrics (where deterioration might be acceptable)
        trade_off_metrics = [
            'noise_variation', 'horizontal_shift', 'oscillation'
        ]
        
        # Define critical metrics (should not significantly deteriorate)
        critical_metrics = [
            'shape_deformation', 'distribution_drift'
        ]
        
        # Analyze each category
        for metric_name, metric_data in all_run_metrics.items():
            if len(metric_data['std']) == 0 or len(metric_data['anti']) == 0:
                continue
            
            # Statistical test
            stat_result = self.stats_analyzer.compute_antifragile_stats(
                metric_data['std'], metric_data['anti'], metric_name
            )
            
            # Categorize result
            if metric_name in antifragile_core_metrics:
                analysis['antifragile_hypothesis_tests'][metric_name] = stat_result
            elif metric_name in trade_off_metrics:
                analysis['trade_off_analysis'][metric_name] = stat_result
            else:
                analysis['conditional_performance'][metric_name] = stat_result
        
        return analysis
    
    def print_antifragile_summary(self, antifragile_analysis: Dict):
        """Print comprehensive antifragile analysis summary."""
        print("\n" + "="*80)
        print("ANTIFRAGILE ANALYSIS SUMMARY")
        print("="*80)
        
        # Core antifragile metrics
        core_metrics = antifragile_analysis.get('antifragile_hypothesis_tests', {})
        print(f"\n📊 CORE ANTIFRAGILE METRICS ({len(core_metrics)} tested):")
        print("-" * 60)
        
        for metric, result in core_metrics.items():
            status = result['antifragile_interpretation']
            improvement = result['improvement_pct']
            p_val = result['p_value']
            print(f"{metric:<25} {status:<30} ({improvement:+6.1f}%, p={p_val:.4f})")
        
        # Trade-off analysis
        trade_metrics = antifragile_analysis.get('trade_off_analysis', {})
        print(f"\n⚖️  TRADE-OFF ANALYSIS ({len(trade_metrics)} tested):")
        print("-" * 60)
        
        for metric, result in trade_metrics.items():
            status = result['antifragile_interpretation']
            improvement = result['improvement_pct']
            p_val = result['p_value']
            print(f"{metric:<25} {status:<30} ({improvement:+6.1f}%, p={p_val:.4f})")
        
        # Overall assessment
        core_successes = sum(1 for m in core_metrics.values() 
                            if m['is_significant'] and m['improvement_pct'] > 0)
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print("-" * 60)
        print(f"Antifragile benefits confirmed: {core_successes}/{len(core_metrics)}")
        print(f"Success rate: {core_successes/max(len(core_metrics), 1)*100:.1f}%")
        
        if core_successes >= len(core_metrics) * 0.5:
            print("✅ CONCLUSION: A-vbnf demonstrates significant antifragile properties")
        elif core_successes > 0:
            print("⚠️  CONCLUSION: A-vbnf shows partial antifragile benefits")
        else:
            print("❌ CONCLUSION: No clear antifragile benefits detected")
        
        print("="*80)

class AntifragileExperimentRunner:
    """Runner for antifragile-focused experiments."""
    
    def __init__(self):
        self.analyzer = AntifragileAnalyzer()
        self.stats_analyzer = StatisticalAnalyzer()
    
    def run_antifragile_focused_tests(self, vis_flg: bool = True, num_runs: int = 20) -> tuple:
        """Enhanced testing framework focused on antifragile hypotheses."""
        import torch
        import numpy as np
        import pyro
        from ..evaluation.stress_tests import StressTester
        from ..data.generators import generate_banana_dataset
        from ..core.training import train_flow, train_flow_antifragile_centered
        import zuko
        
        # Your existing test structure
        all_run_metrics = {}
        
        for i in range(num_runs):
            seed_value = i
            torch.manual_seed(seed_value)
            np.random.seed(seed_value)
            pyro.set_rng_seed(seed_value)
            
            print(f"Run {i+1}/{num_runs}")
            
            # Generate data
            data = generate_banana_dataset(n_nominal=1000, n_target=5, seed=seed_value)
            nominal_data = data['nominal']
            target_data = data['target']
            
            # Train models
            standard_flow = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
            standard_flow, _ = train_flow(
                standard_flow, nominal_data, target_data, 
                context_dim=1, num_steps=800, lr=1e-3, subsets=5, 
                use_antifragile=False
            )
            
            antifragile_flow = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
            antifragile_flow, _ = train_flow_antifragile_centered(
                antifragile_flow, nominal_data, target_data, 
                context_dim=1, num_steps=800, lr=1e-3, subsets=5
            )
            
            # Run stress tests
            stress_tester = StressTester()
            all_results = stress_tester.run_all_stress_tests(
                standard_flow, antifragile_flow, nominal_data, target_data, vis_flg=False
            )
            
            # Extract metrics
            extracted_metrics = self.stats_analyzer.extract_metric_values(all_results)
            
            # Add antifragile-specific metrics
            antifragile_metrics = self.analyzer.calculate_antifragile_metrics(all_results)
            extracted_metrics.update(antifragile_metrics)
            
            # Collect results
            for metric_name, metric_data in extracted_metrics.items():
                if metric_name not in all_run_metrics:
                    all_run_metrics[metric_name] = {'std': [], 'anti': []}
                
                if len(metric_data['std']) > 0:
                    all_run_metrics[metric_name]['std'].append(np.mean(metric_data['std']))
                if len(metric_data['anti']) > 0:
                    all_run_metrics[metric_name]['anti'].append(np.mean(metric_data['anti']))
        
        # Antifragile-focused analysis
        antifragile_analysis = self.analyzer.analyze_antifragile_results(all_run_metrics)
        
        if vis_flg:
            from ..visualization.dashboards import create_antifragile_visualizations
            create_antifragile_visualizations(all_run_metrics, antifragile_analysis)
        
        return all_run_metrics, antifragile_analysis

# Factory functions for backward compatibility
def calculate_antifragile_metrics(all_results: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Factory function for calculating antifragile metrics."""
    analyzer = AntifragileAnalyzer()
    return analyzer.calculate_antifragile_metrics(all_results)

def analyze_antifragile_results(all_run_metrics: Dict[str, Dict[str, List[float]]]) -> Dict:
    """Factory function for analyzing antifragile results."""
    analyzer = AntifragileAnalyzer()
    return analyzer.analyze_antifragile_results(all_run_metrics)