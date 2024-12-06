import numpy as np
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import time
import os

@dataclass
class Instance:
    id: int
    features: np.ndarray
    label: int

class Validator:
    def __init__(self):
        self.classifier = NearestNeighborClassifier()
    
    def load_data(self, filename: str) -> List[Instance]:
        """load and normalize data from file."""
        try:
            data = np.loadtxt(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Count not find the file: {filename}\nPlease check the file path and try again.")
        labels = data[:, 0]
        features = data[:, 1:]

        #normalize features
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

        return [
            Instance(id = i, features=features[i], label=int(labels[i]))
            for i in range(len(labels))
        ]
    
    def evaluate_feature_subset(self, data: List[Instance], feature_subset: Set[int]) -> Tuple[float, float]:
        """Perform leave-one-out cross validation using only specific features.
        returns (accuracy as percentage, time taken)."""
        correct_predictions = 0
        feature_indices = [i - 1 for i in feature_subset] # convert to 0 based indexing

        start_time = time.time()
        #for each instance
        for i, test_instance in enumerate(data):
            #create training set excluding current instance
            training_data = data[:i] + data[i+1:]

            #extract selected features only
            test_features = test_instance.features[feature_indices]
            train_features = [
                Instance(
                    id=inst.id,
                    features=inst.features[feature_indices],
                    label=inst.label
                )
                for inst in training_data
            ]
            #train and test
            self.classifier.train(train_features)
            test_inst = Instance(test_instance.id, test_features, test_instance.label)
            prediction = self.classifier.test(test_inst)

            if prediction == test_instance.label:
                correct_predictions += 1
            accuracy = (correct_predictions / len(data)) * 100
            elapsed_time = time.time() - start_time

            return accuracy, elapsed_time
        
    def get_dataset_path(dataset_name: str) -> str:
        """get dataset path from user input with validation"""
        while True:
            path = input(f"Enter the path to {dataset_name} (or 'q' to quit): ")
            if path.lower() == 'q':
                exit()
            if os.path.exists(path):
                return path
            print(f"File not found at: {path}")
            print("Please check the path and try again.")

def main():
    validator = Validator()
        
    # Here i am settijng some of the printing lines for prompt user
    print("Welcome to the Nearest Neighbor Classifier testing program the prompt will display below:")
    print("\nWe will test the classifier on two datasets with specific feature subsets and they are shown below:")
    print("1. Small dataset using features {3, 5, 7}")
    print("2. Large dataset using features {1, 15, 27}")
        
    # Test on small dataset and it will load the feature as you can see 3 5 7 are out features,
    print("\nTesting on small dataset now downloading and test...")
    small_path = get_dataset_path("small-test-dataset.txt")
    try:
        small_data = validator.load_data(small_path)
        # Load features correctly
        small_features = {3, 5, 7}
        accuracy, time_taken = validator.evaluate_feature_subset(small_data, small_features)
        # Analyse
        print(f"Small dataset accuracy with features {small_features}: {accuracy:.2f}%")
        print(f"Time taken: {time_taken:.2f} seconds\n")
    except Exception as e:
        print(f"Error processing small dataset: {str(e)}")

if __name__ == "__main__":
    main()