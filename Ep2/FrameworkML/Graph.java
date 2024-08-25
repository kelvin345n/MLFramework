package Ep2.FrameworkML;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Used in graphical neural networks.  */
public class Graph <T> {
    // A Hash map that stores that item's adjacency list
    private Map<T, Set<T>> adj_map;

    public Graph(){
        adj_map = new HashMap<>();
    }

    /** Creates a node with no connections if it is not already in the graph */
    public void createNode(T data){
        if (!adj_map.containsKey(data)){
            adj_map.put(data, new HashSet<>());
        }
    }

    /** Creates an edge between item1 and item2 */
    public void addEdge(T item1, T item2){
        // Creates a node if not already in graph
        createNode(item1);
        createNode(item2);

        Set<T> item1Set = adj_map.get(item1);
        Set<T> item2Set = adj_map.get(item2);

        // Adds each other to each adjacency list
        item1Set.add(item2);
        item2Set.add(item1);
    }

    /** Checks if item1 and item2 are connected in the graph */
    public boolean isDirectlyConnected(T item1, T item2){
        if (!adj_map.containsKey(item1) || !adj_map.containsKey(item2)){
            return false;
        } else {
            Set<T> item1Set = adj_map.get(item1);
            return item1Set.contains(item2);
        }
    }

    /** Returns a copy of the neighbors of that item. Null if item does not exist in graph */
    public Set<T> getNeighbors(T item){
        if (adj_map.get(item) == null){
            return null;
        }
        return new HashSet<>(adj_map.get(item));
    }
}