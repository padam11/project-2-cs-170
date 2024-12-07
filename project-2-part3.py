import numpy as np
from typing import List, Set, Tuple
from dataclasses import dataclass
import time

@dataclass
class Instance:
    id: int
    features: np.ndarray
    label: int

class NearestNeighborClassifier:
    def __init__(self):
        self.training_data: List[Instance] = []

    def train(self, instances: List[Instance]):
        self.training_data = instances
    
    def test(self, instance: Instance) -> int:
        if not self.training_data:
            raise ValueError("Classifier must be trained before testing") 
        #compute euclidean distance to find closest training instance to given test instance
        min_distance = float('inf')
        predicted_label = None
        for train_instance in self.training_data:
            distance = np.sqrt(np.sum((instance.features - train_instance.features) ** 2))
            if distance < min_distance: #update if computed dist is smaller than curr min_dist
                min_distance = distance
                predicted_label = train_instance.label
        return predicted_label

class FeatureSelection:
    def __init__(self, data: List[Instance]):
        self.data = data
        self.num_features = len(data[0].features)
        self.classifier = NearestNeighborClassifier()
        
    def evaluate_feature_subset(self, feature_subset: Set[int]) -> float:
        """Perform leave-one-out cross validation using only specified features."""
        if not feature_subset:  # Handle empty feature set
            return 0.0
            
        correct_predictions = 0
        feature_indices = [i - 1 for i in feature_subset]  # Convert to 0-based indexing
        
        # for each instance
        for i, test_instance in enumerate(self.data):
            # create training set excluding current instance
            training_data = self.data[:i] + self.data[i+1:]
            
            # extract selected features only
            test_features = test_instance.features[feature_indices]
            train_features = [
                Instance(
                    id=inst.id,
                    features=inst.features[feature_indices],
                    label=inst.label
                )
                for inst in training_data
            ]
            
            # train and test
            self.classifier.train(train_features)
            test_inst = Instance(test_instance.id, test_features, test_instance.label)
            prediction = self.classifier.test(test_inst)
            
            if prediction == test_instance.label:
                correct_predictions += 1
                
        accuracy = (correct_predictions / len(self.data)) * 100
        return accuracy

    def forward_selection(self) -> Tuple[Set[int], float]:
        """Implements forward selection search algorithm."""
        current_features: Set[int] = set()
        best_score = self.evaluate_feature_subset(current_features)
        best_features = current_features.copy()

        print(f"Using no features, I get an accuracy of {best_score:.1f}%")
        print("\nBeginning search.")

        while len(current_features) < self.num_features:
            candidate_score = -1
            candidate_feature = -1

            for feature in range(1, self.num_features + 1):
                if feature not in current_features:
                    new_features = current_features | {feature}
                    score = self.evaluate_feature_subset(new_features)
                    
                    print(f"Using feature(s) {sorted(new_features)} accuracy is {score:.1f}%")

                    if score > candidate_score:
                        candidate_score = score
                        candidate_feature = feature

            print(f"Feature set {sorted(current_features | {candidate_feature})} was best, accuracy is {candidate_score:.1f}%")
            
            current_features.add(candidate_feature)

            if candidate_score > best_score:
                best_score = candidate_score
                best_features = current_features.copy()
            elif len(current_features) < self.num_features:
                print("(Warning, Accuracy has decreased!)")

        print(f"\nFinished search!!")
        print(f"The best feature subset is {sorted(best_features)}, which has an accuracy of {best_score:.1f}%")
        
        return best_features, best_score

    def backward_elimination(self) -> Tuple[Set[int], float]:
        """Implements backward elimination search algorithm."""
        current_features = set(range(1, self.num_features + 1))
        best_score = self.evaluate_feature_subset(current_features)
        best_features = current_features.copy()
        
        print(f"Using all features, I get an accuracy of {best_score:.1f}%")
        print("\nBeginning search.")

        while current_features:
            candidate_score = -1
            candidate_feature = -1
            #score: accuracy score of the new feature subset.
            for feature in current_features:
                new_features = current_features - {feature}
                score = self.evaluate_feature_subset(new_features)
                
                print(f"Using feature(s) {sorted(new_features)} accuracy is {score:.1f}%")
                # if new subset score is higher than curr best candidate score
                if score > candidate_score:
                    candidate_score = score
                    candidate_feature = feature
            
            print(f"Feature set {sorted(current_features - {candidate_feature})} was best, accuracy is {candidate_score:.1f}%")
            #remove candidate feature from curr features because it results in best
            #performance when moved
            current_features.remove(candidate_feature)
            #update global best_score and best_features
            if candidate_score > best_score:
                best_score = candidate_score
                best_features = current_features.copy()
            elif current_features:
                print("(Warning, Accuracy has decreased!)")
                
        print(f"\nFinished search!!")
        print(f"The best feature subset is {sorted(best_features)}, which has an accuracy of {best_score:.1f}%")
        
        return best_features, best_score

def load_data(filename: str) -> List[Instance]:
    """Load and normalize data from file."""
    try:
        data = np.loadtxt(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {filename}")
        
    labels = data[:, 0] #first column is the label
    features = data[:, 1:] #everything else are the features
    
    # Normalize features
    #subtracts the mean of each feature column (np.mean(features, axis=0))
    #divides each column by its standard deviation
    #ensures all features have a mean of 0 and a standard dev of 1
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    #creates/returns an instance object with id, normalized features, and label
    return [
        Instance(id=i, features=features[i], label=int(labels[i]))
        for i in range(len(labels))
    ]

def main():
    print("Welcome to Feature Selection")
    
    # Get filename from user
    filename = input("\nType in the name of the file to test: ")
    
    try:
        # ;oad and preprocess data
        data = load_data(filename)
        num_features = len(data[0].features)
        print(f"\nThis dataset has {num_features} features (not including the class attribute), with {len(data)} instances.")
        
        # initialize feature selection
        feature_selection = FeatureSelection(data)
        
        # get algorithm choice
        print("\nType the number of the algorithm you want to run:")
        print("1. Forward Selection")
        print("2. Backward Elimination")
        
        try:
            choice = int(input("Enter your choice: "))
            
            if choice == 1:
                start_time = time.time()
                best_features, best_score = feature_selection.forward_selection()
                elapsed_time = time.time() - start_time
                print(f"\nTime taken: {elapsed_time:.2f} seconds")
            elif choice == 2:
                start_time = time.time()
                best_features, best_score = feature_selection.backward_elimination()
                elapsed_time = time.time() - start_time
                print(f"\nTime taken: {elapsed_time:.2f} seconds")
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()