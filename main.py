import argparse
import sys
from config import get_config
from experiments.banana_experiment import run_banana_experiment
from experiments.stress_experiments import run_all_stress_tests, run_multiple_experiments
from analysis.antifragile_analysis import AntifragileExperimentRunner

def main():
    """Main entry point for the antifragile vbnf experiments."""
    parser = argparse.ArgumentParser(description='Antifragile vbnf Experiments')
    
    parser.add_argument('--experiment', type=str, default='banana',
                       choices=['banana', 'stress_test', 'antifragile_focused', 'multiple_runs'],
                       help='Type of experiment to run')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration name or path to config file')
    
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs for statistical analysis')
    
    parser.add_argument('--vis', action='store_true',
                       help='Enable visualizations')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        if args.config.endswith('.json'):
            from config import ExperimentConfig
            config = ExperimentConfig.load(args.config)
        else:
            config = get_config(args.config)
    else:
        # Use default config based on experiment type
        if args.experiment in ['banana']:
            config = get_config('banana')
        elif args.experiment in ['stress_test', 'multiple_runs']:
            config = get_config('stress_test')
        elif args.experiment == 'antifragile_focused':
            config = get_config('banana')  # Use banana as base
        else:
            config = get_config('quick_test')
    
    print(f"Running experiment: {args.experiment}")
    print(f"Configuration: {config.name}")
    
    # Run the appropriate experiment
    if args.experiment == 'banana':
        results = run_banana_experiment(config, device)
        print("Banana experiment completed successfully!")
        
    elif args.experiment == 'stress_test':
        results = run_all_stress_tests(config, args.vis, device)
        print("Stress test completed successfully!")
        
    elif args.experiment == 'multiple_runs':
        all_metrics, stats_results = run_multiple_experiments(
            args.num_runs, config, save_results=True, device=device
        )
        print(f"Multiple runs experiment ({args.num_runs} runs) completed successfully!")
        
    elif args.experiment == 'antifragile_focused':
        runner = AntifragileExperimentRunner()
        all_metrics, antifragile_analysis = runner.run_antifragile_focused_tests(
            vis_flg=args.vis, num_runs=args.num_runs
        )
        
        # Print comprehensive summary
        runner.analyzer.print_antifragile_summary(antifragile_analysis)
        print(f"Antifragile-focused experiment ({args.num_runs} runs) completed successfully!")
    
    print("\nExperiment completed! Check the generated files and visualizations.")

if __name__ == "__main__":
    main()


