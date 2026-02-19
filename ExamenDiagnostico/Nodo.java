package arbol;

public class Nodo {
    String nombre;
    Nodo izq;
    Nodo dere;

    public Nodo(String nombre) {
        this.nombre = nombre;
        this.izq = null;
        this.dere = null;
    }
}
