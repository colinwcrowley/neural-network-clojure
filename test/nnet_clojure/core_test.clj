(ns nnet-clojure.core-test
  (:require [clojure.test :refer :all]
            [nnet-clojure.core :refer :all]))

(deftest dementionality-reduction
  (testing "Dementionality reduction 1-4 to binary."
    (let [network (build-net "encoder-net" [4 2 4] #(/ (- (rand) 0.5) 20) 8) ;4 inputs and outputs and 2 in the hidden layer. learning rate = 8
          training-set {:name "encoder-training-set"
                        :examples '({:input [0 0 0 0] :output [0.01 0.01 0.01 0.01]}
                                    {:input [0 0 0 1] :output [0.01 0.01 0.01 0.95]}
                                    {:input [0 0 1 0] :output [0.01 0.01 0.95 0.01]}
                                    {:input [0 1 0 0] :output [0.01 0.95 0.01 0.01]}
                                    {:input [1 0 0 0] :output [0.95 0.01 0.01 0.01]})}
          trained-net (train network training-set)] ;200 training epochs
      (is (= 1 (evaluate-performance trained-net training-set))))))
