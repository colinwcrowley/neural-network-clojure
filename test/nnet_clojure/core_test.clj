(ns nnet-clojure.core-test
  (:require [clojure.test :refer :all]
            [nnet-clojure.core :refer :all]
            [clojure.string :as str]))

(deftest dementionality-reduction
  (testing "Dementionality reduction 1-4 to binary."
    (let [network (build-net "encoder-net" [4 2 4] #(/ (- (rand) 0.5) 20) 8) ;4 inputs and outputs and 2 in the hidden layer. learning rate = 8
          training-set {:name "encoder-training-set"
                        :examples '({:input [0 0 0 0] :output [0.01 0.01 0.01 0.01]}
                                    {:input [0 0 0 1] :output [0.01 0.01 0.01 0.95]}
                                    {:input [0 0 1 0] :output [0.01 0.01 0.95 0.01]}
                                    {:input [0 1 0 0] :output [0.01 0.95 0.01 0.01]}
                                    {:input [1 0 0 0] :output [0.95 0.01 0.01 0.01]})}
          trained-net (train network training-set training-set)] ;200 training epochs
      (is (= 1 (evaluate-performance trained-net training-set))))))

(deftest digit-classifiction
  (testing "training a network to classify hand written digits"
    (letfn [(read-to-listvec
              [filename]
              (->> filename
                   (slurp)
                   (#(str/split % #"\n"))
                   (map #(str/split % #" "))
                   (map #(map read-string %))
                   (map #(map double %))
                   (map vec)))
            (read-to-dataset
              [input-filename output-filename dataset-name]
              {:name dataset-name
               :examples (apply list (map (fn [in out] {:input in :output out})
                                          (read-to-listvec input-filename)
                                          (read-to-listvec output-filename)))})
            (split-dataset
              [dataset training-set-name test-set-name fence]
              [{:name training-set-name
                :examples (apply list (take fence (:examples dataset)))}
               {:name test-set-name
                :examples (apply list (drop fence (:examples dataset)))}])]
      (let [network (build-net "digit-net" [64 10 10] #(/ (- (rand) 0.5) 20) 2)
            datasets (split-dataset (read-to-dataset "resources/digit-inputs.dat" "resources/digit-targets.dat" "")
                                    "digit-training-set"
                                    "digit-test-set" 60)
            trained-net (train network (first datasets) (second datasets))]
        (is (> (evaluate-performance trained-net (second datasets)) 20))))))


     ;list of input vectors
