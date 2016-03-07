(ns nnet-clojure.core
  (:require [loom.graph :as gr]
            [loom.io :as io])
  (:gen-class))

(declare evaluate-input get-activations build-net
         train epoch updated-network updated-node-weights
         delta cross sigmoid get-node-id evaluate-performance
         similar? training-msg)

(defn -main
  "I don't do a whole lot ... yet."
  [& args])

(defn evaluate-performance
  "returns the fraction of the examples that the network classified correctly"
  ([network data-set equivalence-func]
   (/ (reduce + (map #(if (equivalence-func (:output %) (evaluate-input network (:input %))) 1 0) (:examples data-set)))
      (count (:examples data-set))))
  ([network data-set]
   (evaluate-performance network data-set similar?)))

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
  [net-name nodes-in-layers weight-init-func learning-rate]
  (let [layers (map #(take % (repeatedly get-node-id)) nodes-in-layers)
        bias-node (get-node-id)
        intra-layer-edges (reduce into (map cross layers (rest layers)))
        bias-edges (map (partial vector bias-node) (reduce into (rest layers)))
        graph (apply gr/weighted-digraph (map #(conj % (weight-init-func)) (into intra-layer-edges bias-edges)))]
    {:name net-name
     :graph graph
     :layers layers
     :bias-node bias-node
     :learning-rate learning-rate}))

(defn train
  "just backpropagation"
  ([network data-set epochs]
   (training-msg network data-set)
   (loop [network network
          epochs-left epochs]
     (println "epochs left: " epochs-left)
     (if (> epochs-left 0)
       (recur (epoch network (:examples data-set)) (dec epochs-left))
       network)))
  ([network data-set]
   (training-msg network data-set)
   (loop [network network
          performance (evaluate-performance network data-set)]
     (println "performance:" performance)
     (if (= performance 1)
       network
       (recur (epoch network (:examples data-set))
              (evaluate-performance network data-set))))))


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
             (if (some (partial = node) (last (:layers network)))
               (* (- (activations node) (output-targets node)) (activations node) (- 1 (activations node)))
               (* (activations node)
                  (- 1 (activations node))
                  (reduce + (map #(* (delta network activations output-targets (second %))
                                     (gr/weight (:graph network) (first %) (second %)))
                                 (gr/out-edges (:graph network) node))))))))

(defn training-msg
  [network data-set]
   (println (str "\n" (apply str (take 35 (repeat "#")))
                 "\nTraining..."
                 "\n  Network: " (:name network)
                 "\n  Data Set: " (:name data-set)
                 "\n" (apply str (take 35 (repeat "#"))))))


(defn similar?
  [vec1 vec2]
  (= (map #(Math/round %) vec1) (map #(Math/round %) vec2)))

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

; (defmacro prret
;   "short for print-and-return. a macro I used for debuging"
;   [form]
;   `(do (println ~form) ~form))
