# analysis.py
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(results_dir):
    # Load all result files from directory
    result_files = glob.glob(os.path.join(results_dir, 'results_*.json'))
    all_results = []
    
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    
    return all_results

def plot_attack_comparison(results):
    # Plot comparison of different attack types
    attack_types = set()
    metrics = {}
    
    # Organize data by attack type
    for result in results:
        attack_type = result['parameters']['attack']
        attack_types.add(attack_type)
        
        if attack_type not in metrics:
            metrics[attack_type] = {
                'iterations': [],
                'percentage_increase': [],
                'final_boxes': []
            }
        
        # Get the final metrics
        final = result['final_metrics']
        metrics[attack_type]['iterations'].append(final['iterations'])
        metrics[attack_type]['percentage_increase'].append(final['percentage_increase'])
        metrics[attack_type]['final_boxes'].append(final['current_boxes'])
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Percentage Increase
    plt.subplot(1, 3, 1)
    for attack in attack_types:
        plt.bar(attack, np.mean(metrics[attack]['percentage_increase']), 
                yerr=np.std(metrics[attack]['percentage_increase']),
                label=attack)
    plt.title('Percentage Increase in Detections')
    plt.ylabel('Percentage Increase')
    
    # Final Box Count
    plt.subplot(1, 3, 2)
    for attack in attack_types:
        plt.bar(attack, np.mean(metrics[attack]['final_boxes']), 
                yerr=np.std(metrics[attack]['final_boxes']),
                label=attack)
    plt.title('Final Detection Count')
    plt.ylabel('Number of Boxes')
    
    # Iterations to Max Effect
    plt.subplot(1, 3, 3)
    for attack in attack_types:
        plt.bar(attack, np.mean(metrics[attack]['iterations']), 
                yerr=np.std(metrics[attack]['iterations']),
                label=attack)
    plt.title('Iterations to Maximum Effect')
    plt.ylabel('Iterations')
    
    plt.tight_layout()
    plt.legend()
    plt.savefig('attack_comparison.png')
    plt.show()

if __name__ == '__main__':
    results = load_results('results')
    plot_attack_comparison(results)