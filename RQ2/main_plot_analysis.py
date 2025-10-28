import os
import sys
from analysis import generate_summary, plot_wr_cs, plot_sw, plot_gini, plot_violin
from analysis.compare_all import compare_all_experiments
from analysis.plot_sw_ratio import plot_sw_ratio


def process_experiment_directory(exp_dir, out_root):
    """
    Process a single experiment directory using existing CSV files to generate plots
    """
    exp_name = os.path.basename(exp_dir)
    print(f"🔍 Processing experiment: {exp_name}")
    
    # Build file paths
    consumer_csv = os.path.join(exp_dir, "consumer_temp.csv")
    firm_csv = os.path.join(exp_dir, "firm_temp.csv")
    data_csv = os.path.join(exp_dir, "data.csv")
    gini_csv = os.path.join(exp_dir, "gini.csv")
    
    # Check required files exist
    required_files = [consumer_csv, firm_csv, data_csv, gini_csv]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️  Warning: missing required files: {[os.path.basename(f) for f in missing_files]}")
        return False
    
    # Create output directory
    out_dir = os.path.join(out_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        # Copy CSV files to output directory
        print("📋 Copying data files...")
        import shutil
        shutil.copy2(consumer_csv, os.path.join(out_dir, "consumer_temp.csv"))
        shutil.copy2(firm_csv, os.path.join(out_dir, "firm_temp.csv"))
        shutil.copy2(data_csv, os.path.join(out_dir, "data.csv"))
        shutil.copy2(gini_csv, os.path.join(out_dir, "gini.csv"))
        
        # Copy consumer folder if exists
        consumer_dir = os.path.join(exp_dir, "consumer")
        if os.path.exists(consumer_dir):
            out_consumer_dir = os.path.join(out_dir, "consumer")
            if os.path.exists(out_consumer_dir):
                shutil.rmtree(out_consumer_dir)
            shutil.copytree(consumer_dir, out_consumer_dir)
            print(f"   ✅ Copied consumer folder")
        
        # Generate plots
        print("📈 Generating visualizations...")
        
        # Plot using existing data.csv
        plot_wr_cs(data_csv, out_dir)
        plot_sw(data_csv, out_dir)
        # New: plot ratios vs BNE
        plot_sw_ratio(data_csv, out_dir)
        
        # Plot using existing CSVs
        plot_gini(consumer_csv, firm_csv, out_dir)
        plot_violin(out_dir, out_dir)
        
        print(f"✅ Completed: {exp_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    # Support CLI arg: python main_plot_analysis.py <root_dir>
    if len(sys.argv) > 1 and sys.argv[1].strip():
        root_dir = sys.argv[1].strip()
        print(f"[INFO] Using CLI arg root_dir={root_dir}")
    else:
        # Backward compatible interactive input
        root_dir = input("Enter root directory containing experiment folders (e.g., all_results_old): ").strip()
    
    # Set output root directory
    out_root = "plot_analysis_results"
    os.makedirs(out_root, exist_ok=True)
    
    print(f"🚀 Start plotting analysis...")
    print(f"📂 Input directory: {root_dir}")
    print(f"📂 Output directory: {out_root}")
    print("=" * 80)
    
    # Stats
    total_dirs = 0
    success_count = 0
    failed_dirs = []
    
    # Check root exists
    if not os.path.exists(root_dir):
        print(f"❌ Error: directory {root_dir} does not exist")
        return
    
    # Iterate subdirectories in root
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        
        # Only handle directories, skip special ones
        if os.path.isdir(item_path) and item not in ["pairwise_tests", "reports"]:
            total_dirs += 1
            
            # Check required CSV files
            required_files = [
                os.path.join(item_path, "consumer_temp.csv"),
                os.path.join(item_path, "firm_temp.csv"),
                os.path.join(item_path, "data.csv"),
                os.path.join(item_path, "gini.csv")
            ]
            
            if all(os.path.exists(f) for f in required_files):
                if process_experiment_directory(item_path, out_root):
                    success_count += 1
                else:
                    failed_dirs.append(item)
            else:
                print(f"⏭️  Skip {item}: missing required CSV files")
                failed_dirs.append(item)
    
    print("=" * 80)
    print("📋 Processing summary:")
    print(f"   - Total directories: {total_dirs}")
    print(f"   - Successfully processed: {success_count}")
    print(f"   - Failed/Skipped: {len(failed_dirs)}")
    
    if failed_dirs:
        print(f"   - Failed directories: {failed_dirs}")
    
    # Run pairwise tests after all experiments processed
    if success_count > 0:
        print("\n🔬 Starting pairwise tests...")
        try:
            compare_all_experiments(out_root)
            print("✅ Pairwise tests completed")
        except Exception as e:
            print(f"❌ Pairwise tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 All plotting analysis completed!")
    print(f"📁 Results saved to: {out_root}")


if __name__ == "__main__":
    main()

