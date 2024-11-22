import random
from typing import Set, List, Tuple

class FeatureSelection:
    def __init__(self, num_features: int):
        self.num_features = num_features #controls the num of features
    def random_evaluation(self, feature_subset: Set[int]) -> float:
        return random.uniform(0, 100) #as requested, dummy eval function that returns random score
    
    def forward_selection(self) -> Tuple[Set[int], float]:
        """
        implements forward selection search algorithm. 
        Returns: (best_features, best_score)
        """
        current_features: Set[int] = set()
        best_score = self.random_evaluation(current_features) #random eval
        best_features = current_features.copy()

        print(f"Using no features and \"random\" evaluation, I get an accuracy of {best_score:.1f}%")
        print("\nBeginning search.")

        #while we're still able to add features
        while len(current_features) < self.num_features:
            candidate_score = -1
            candidate_feature = -1

            #add each unused feature
            for feature in range(1, self.num_features + 1):
                if feature not in current_features:
                    new_features = current_features | {feature}
                    score = self.random_evaluation(new_features)

                    print(f"Using feature(s) {sorted(new_features)} accuracy is {score:.1f}%")

                    if score > candidate_score:
                        candidate_score = score
                        candidate_feature = feature
                    
            print(f"Feature set {sorted(current_features | {candidate_feature})} was best, accuracy is {candidate_score:.1f}%")

            #add best feature found
            current_features.add(candidate_feature)

            #update best socre if current set is better
            if candidate_score > best_score:
                best_score = candidate_score
                best_features = current_features.copy()
            elif len(current_features) < self.num_features:
                print("(Warning, Accuracy has decreased!)")
        print(f"\nFinished search!!")
        print(f"The best feature subset is {sorted(best_features)}, which has an accuracy of {best_score:.1f}%")
        
        return best_features, best_score

    def backward_elimination(self) -> Tuple[Set[int], float]:
        """
        Implements backward elimination search algorithm.
        Returns: (best_features, best_score)
        """
        #start with all features
        current_features = set(range(1, self.num_features + 1))
        best_score = self.random_evaluation(current_features)
        best_features = current_features.copy()
        
        print(f"Using all features and \"random\" evaluation, I get an accuracy of {best_score:.1f}%")
        print("\nBeginning search.")

        
        

