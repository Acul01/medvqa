"""Example usage of the Medical VQA loader in MC mode."""

from utils.load_data import MedicalVQALoader


def main():
    from utils.paths import get_data_root, get_mc_data_root

    loader = MedicalVQALoader(data_root=get_data_root(), mc_data_root=get_mc_data_root())

    print("=" * 60)
    print("MC Loader Example")
    print("=" * 60)
    print(f"Available datasets: {loader.get_available_datasets()}")

    samples = loader.load_dataset("pathvqa", split="train")
    print(f"Loaded PathVQA train MC samples: {len(samples)}")

    if samples:
        s = samples[0]
        print("\nFirst sample")
        print(f"  image_path: {s.image_path}")
        print(f"  question: {s.question}")
        print(f"  options: {s.mc_options}")
        print(f"  correct_index: {s.mc_correct_index}")
        print(f"  ground_truth_letter: {s.answer}")
        print(f"  question_type: {s.question_type}")
        print(f"  ground_truth_text: {(s.metadata or {}).get('gt_answer_text', '')}")

    combined_val = loader.load_combined(split="validation", vqarad_split="test")
    print(f"\nCombined validation samples (incl. Med-VQA-2019): {len(combined_val)}")


if __name__ == "__main__":
    main()
