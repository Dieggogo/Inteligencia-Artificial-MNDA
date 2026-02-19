package arbol;

public class Arbol {
    private Nodo raiz;

    public Arbol() {
        this.raiz = null;
    }
    public boolean vacio() {
        return raiz == null;
    }
    public void insertar(String nombre) {
        raiz = insertarRec(raiz, nombre);
    }

    private Nodo insertarRec(Nodo actual, String nombre) {
        if (actual == null) return new Nodo(nombre);

        if (nombre.compareToIgnoreCase(actual.nombre) < 0) {
            actual.izq = insertarRec(actual.izq, nombre);
        } else if (nombre.compareToIgnoreCase(actual.nombre) > 0) {
            actual.dere = insertarRec(actual.dere, nombre);
        }
        return actual;
    }
    public Nodo buscarNodo(String nombre) {
        return buscarPreorden(raiz, nombre);
    }

    private Nodo buscarPreorden(Nodo actual, String nombre) {
        if (actual == null) return null;

        if (actual.nombre.equalsIgnoreCase(nombre)) {
            return actual;
        }
        Nodo encontradoIzq = buscarPreorden(actual.izq, nombre);
        if (encontradoIzq != null) return encontradoIzq;

        return buscarPreorden(actual.dere, nombre);
    }
}
