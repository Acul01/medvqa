"""
Example usage of the Medical VQA Data Loader

This script demonstrates how to use the unified loader for PathVQA, SLAKE, and VQA-Rad datasets.
"""

from utils.load_data import MedicalVQALoader, VQASample, load_pathvqa, load_slake, load_vqarad


def example_basic_usage():
    """Basic usage examples"""
    print("=" * 60)
    print("Basic Usage Examples")
    print("=" * 60)
    
    # Initialize loader
    loader = MedicalVQALoader(data_root="data")
    
    # Check available datasets
    print(f"\nAvailable datasets: {loader.get_available_datasets()}")
    
    # Check available splits for each dataset
    for dataset in loader.get_available_datasets():
        splits = loader.get_available_splits(dataset)
        print(f"{dataset} splits: {splits}")
    
    print("\n" + "-" * 60)


def example_load_single_split():
    """Load a single split from a dataset"""
    print("\nLoading single splits...")
    
    loader = MedicalVQALoader(data_root="data")
    
    # Load PathVQA train split
    try:
        pathvqa_train = loader.load_dataset("PathVQA", split="train")
        print(f"\nPathVQA train: {len(pathvqa_train)} samples")
        if pathvqa_train:
            print(f"  First sample:")
            print(f"    Image: {pathvqa_train[0].image_path}")
            print(f"    Question: {pathvqa_train[0].question[:80]}...")
            print(f"    Answer: {pathvqa_train[0].answer}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
    
    # Load SLAKE test split
    try:
        slake_test = loader.load_dataset("SLAKE", split="test")
        print(f"\nSLAKE test: {len(slake_test)} samples")
        if slake_test:
            print(f"  First sample:")
            print(f"    Image: {slake_test[0].image_path}")
            print(f"    Question: {slake_test[0].question}")
            print(f"    Answer: {slake_test[0].answer}")
            if slake_test[0].metadata:
                print(f"    Metadata keys: {list(slake_test[0].metadata.keys())[:5]}...")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
    
    # Load VQA-Rad train split
    try:
        vqarad_train = loader.load_dataset("VQA-Rad", split="train")
        print(f"\nVQA-Rad train: {len(vqarad_train)} samples")
        if vqarad_train:
            print(f"  First sample:")
            print(f"    Image: {vqarad_train[0].image_path}")
            print(f"    Question: {vqarad_train[0].question}")
            print(f"    Answer: {vqarad_train[0].answer}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")


def example_load_all_splits():
    """Load all splits from a dataset"""
    print("\n" + "-" * 60)
    print("Loading all splits...")
    
    loader = MedicalVQALoader(data_root="data")
    
    # Load all PathVQA splits
    try:
        pathvqa_all = loader.load_all_splits("PathVQA")
        print(f"\nPathVQA:")
        for split, samples in pathvqa_all.items():
            print(f"  {split}: {len(samples)} samples")
    except FileNotFoundError as e:
        print(f"  Error: {e}")


def example_convenience_functions():
    """Use convenience functions"""
    print("\n" + "-" * 60)
    print("Using convenience functions...")
    
    try:
        # Using convenience functions
        pathvqa_val = load_pathvqa(data_root="data", split="validation")
        print(f"\nPathVQA validation (via convenience function): {len(pathvqa_val)} samples")
        
        slake_train = load_slake(data_root="data", split="train")
        print(f"SLAKE train (via convenience function): {len(slake_train)} samples")
        
        vqarad_test = load_vqarad(data_root="data", split="test")
        print(f"VQA-Rad test (via convenience function): {len(vqarad_test)} samples")
    except FileNotFoundError as e:
        print(f"  Error: {e}")


def example_iterate_samples():
    """Iterate through samples"""
    print("\n" + "-" * 60)
    print("Iterating through samples...")
    
    loader = MedicalVQALoader(data_root="data")
    
    try:
        samples = loader.load_dataset("PathVQA", split="train")
        print(f"\nIterating through first 3 PathVQA train samples:")
        for i, sample in enumerate(samples[:3]):
            print(f"\n  Sample {i+1}:")
            print(f"    Dataset: {sample.dataset}")
            print(f"    Split: {sample.split}")
            print(f"    Image: {sample.image_path}")
            print(f"    Question: {sample.question[:100]}...")
            print(f"    Answer: {sample.answer}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    example_basic_usage()
    example_load_single_split()
    example_load_all_splits()
    example_convenience_functions()
    example_iterate_samples()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
