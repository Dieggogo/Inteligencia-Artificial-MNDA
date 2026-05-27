package puzzle8;

import java.util.LinkedList;

public class Estado {

    public String estado;
    public Estado anterior;
    public int profundidad;
    public int costo;

    public Estado(String estado) {
        this(estado, 0, null);
    }

    public Estado(String estado, int profundidad, Estado anterior) {
        this.estado = normalizarEstado(estado);
        this.profundidad = profundidad;
        this.anterior = anterior;
        this.costo = profundidad;
    }

    public LinkedList<Estado> expandir() {
        LinkedList<Estado> siguientes = new LinkedList<>();

        int indiceVacio = this.estado.indexOf(' ');
        int siguienteProfundidad = this.profundidad + 1;

        switch (indiceVacio) {
            case 0:
                siguientes.add(new Estado(intercambiar(this.estado, 0, 1), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 0, 3), siguienteProfundidad, this));
                break;

            case 1:
                siguientes.add(new Estado(intercambiar(this.estado, 1, 0), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 1, 2), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 1, 4), siguienteProfundidad, this));
                break;

            case 2:
                siguientes.add(new Estado(intercambiar(this.estado, 2, 1), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 2, 5), siguienteProfundidad, this));
                break;

            case 3:
                siguientes.add(new Estado(intercambiar(this.estado, 3, 0), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 3, 4), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 3, 6), siguienteProfundidad, this));
                break;

            case 4:
                siguientes.add(new Estado(intercambiar(this.estado, 4, 1), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 4, 3), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 4, 5), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 4, 7), siguienteProfundidad, this));
                break;

            case 5:
                siguientes.add(new Estado(intercambiar(this.estado, 5, 2), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 5, 4), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 5, 8), siguienteProfundidad, this));
                break;

            case 6:
                siguientes.add(new Estado(intercambiar(this.estado, 6, 3), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 6, 7), siguienteProfundidad, this));
                break;

            case 7:
                siguientes.add(new Estado(intercambiar(this.estado, 7, 4), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 7, 6), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 7, 8), siguienteProfundidad, this));
                break;

            case 8:
                siguientes.add(new Estado(intercambiar(this.estado, 8, 5), siguienteProfundidad, this));
                siguientes.add(new Estado(intercambiar(this.estado, 8, 7), siguienteProfundidad, this));
                break;
        }

        return siguientes;
    }

    public void mostrarRuta() {
        if (this.anterior != null) {
            this.anterior.mostrarRuta();
        }

        System.out.println("Tablero:");
        for (int i = 0; i < 9; i++) {
            System.out.print(this.estado.charAt(i) + " ");
            if ((i + 1) % 3 == 0) System.out.println();
        }

        System.out.println();
        System.out.println("Nivel: " + this.profundidad);
        System.out.println("________________________________");
        System.out.println();
    }

    private String intercambiar(String s, int i, int j) {
        char[] arr = s.toCharArray();
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
        return new String(arr);
    }

    private static String normalizarEstado(String s) {
        if (s == null) return "";

        s = s.replace('0', ' ');

        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < s.length(); k++) {
            char c = s.charAt(k);
            if ((c >= '1' && c <= '8') || c == ' ') sb.append(c);
        }

        if (sb.length() == 9) return sb.toString();
        return s;
    }
}
