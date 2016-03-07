(ns nnet-clojure.core
  (:require [loom.graph :as g]
            [loom.attr :as atr]
            [loom.alg :as alg]
            [loom.io :as io])
  (:gen-class))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (def network (build-net [4 2 4] #(/ (- (rand) 0.5) 10) 2))
  (def examples '({:input [0 0 0 0] :output [0.1 0.1 0.1 0.1]}
                  {:input [0 0 0 1] :output [0.1 0.1 0.1 0.9]}
                  {:input [0 0 1 0] :output [0.1 0.1 0.9 0.1]}
                  {:input [0 1 0 0] :output [0.1 0.9 0.1 0.1]}
                  {:input [1 0 0 0] :output [0.9 0.1 0.1 0.1]}))
  (def trained-net (train network examples 1000)))

(defn evaluate-input
  "returns the activations of the output layer"
  [network input]
  (let [input-map (into {} (map vector (first (:layers network)) input))]
    (vec (map (get-activations network input-map) (last (:layers network))))))

(defn get-activations
  "sends the input map through the network and returns a map of nodes to their activations"
  [network input]
  (loop [unvisited (reverse (apply list (reduce into (rest (:layers network)))))
         current-node (first unvisited)
         activations (into {} (conj input [(:bias-node network) 1]))]
    (if (not (empty? unvisited))
      (recur (pop unvisited)
             (second unvisited)
             (assoc activations current-node
                    (sigmoid
                      (reduce +
                              (map (fn [coll] (* (gr/weight (:graph network) (first coll) current-node) (activations (first coll))))
                                   (gr/in-edges (:graph network) current-node))))))
      activations)))

(defn build-net
  "creates the structure of the graph"
  [nodes-in-layers weight-init-func learning-rate]
  (let [layers (map #(take % (repeatedly get-node-id)) nodes-in-layers)
        bias-node (get-node-id)
        intra-layer-edges (reduce into (map cross layers (rest layers)))
        bias-edges (map (partial vector bias-node) (reduce into (rest layers)))
        graph (apply gr/weighted-digraph (map #(conj % (weight-init-func)) (into intra-layer-edges bias-edges)))]
    {:graph graph
     :layers layers
     :bias-node bias-node
     :learning-rate learning-rate}))

(defn train
  "just backpropagation"
  [network examples epochs]
  (loop [network network
         epochs-left epochs]
    (if (> epochs-left 0)
      (recur (epoch network examples) (dec epochs-left))
      network)))

(defn epoch
  [network examples]
  (loop [network network
         examples-left examples
         current-example (first examples-left)]
    (if (not (empty? examples-left))
      (let [input-map (into {} (map vector (first (:layers network)) (:input current-example)))
            output-map (into {} (map vector (last (:layers network)) (:output current-example)))]
        (recur (updated-network network
                                     (get-activations network input-map)
                                     input-map
                                     output-map)
               (pop examples-left)
               (second examples-left)))
      network)))

(defn updated-network ; Get it working synchronously first, then worry about descending nodes in the same layer in parallel
  [network activations input output]
  (loop [layers-left (reverse (rest (:layers network)))
         current-layer (first layers-left)
         edges-to-change {}]
    (if (not (empty? layers-left))
      (recur (pop layers-left)
             (second layers-left)
             (reduce into edges-to-change
                     (map (partial updated-node-weights network activations input output)
                          current-layer)))
      (assoc network :graph (apply gr/add-edges
                                   (:graph network)
                                   (map conj (gr/edges (:graph network))
                                   (map edges-to-change (gr/edges (:graph network)))))))))

(defn updated-node-weights
  "returns a map from edges to weights after running one gradient decent step"
  [network activations inputs output-targets node]
  (into {} (map vector
                (gr/in-edges (:graph network) node)
                (map (fn [edge] (- (gr/weight (:graph network) (first edge) (second edge))
                                   (* (:learning-rate network)
                                      (delta network activations output-targets (second edge))
                                      (activations (first edge)))))
                     (gr/in-edges (:graph network) node)))))

(def delta
  (memoize (fn [network activations output-targets node]
             ; (println "some result: " (some node (last (:layers network))))
             (if (some (partial = node) (last (:layers network)))
               (* (- (activations node) (output-targets node)) (activations node) (- 1 (activations node)))
               (* (activations node)
                  (- 1 (activations node))
                  (reduce + (map #(* (delta network activations output-targets (second %))
                                     (gr/weight (:graph network) (first %) (second %)))
                                 (gr/out-edges (:graph network) node))))))))

(defn cross
  "returns one set cross another"
  [set1 set2]
  (reduce into (map #(map (partial vector %1) set2) set1)))

(defn sigmoid
  "sigmoid function"
  [x]
  (/ 1 (+ 1 (Math/exp (* -1 x)))))

(def current-node-id (atom 0))
(defn get-node-id
  "unique node ids"
  []
  (do
    (swap! current-node-id inc)
    @current-node-id))

(defmacro par
  "short for print-and-return. a macro I used for debuging"
  [form]
  `(do (println ~form) ~form))
