import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Nodo {
    public static final int LADO = 5;
    public static final int TOTAL_CELDAS = LADO * LADO;
    private static final int BLANCO = 0;

    public enum TipoHeuristica {
        MANHATTAN,
        CONFLICTO_LINEAL
    }

    private final int[] estado;
    private final int indiceBlanco;
    private String clave;

    private Nodo padre;
    private int costo;
    private char movimiento;

    public Nodo(int[] estado) {
        if (estado == null || estado.length != TOTAL_CELDAS) {
            throw new IllegalArgumentException("El estado debe tener 25 valores.");
        }
        this.estado = Arrays.copyOf(estado, estado.length);
        this.indiceBlanco = buscarIndiceBlanco(this.estado);
        this.padre = null;
        this.costo = 0;
        this.movimiento = 'S';
    }

    public static int[] objetivo() {
        int[] objetivo = new int[TOTAL_CELDAS];
        for (int i = 0; i < TOTAL_CELDAS - 1; i++) {
            objetivo[i] = i + 1;
        }
        objetivo[TOTAL_CELDAS - 1] = BLANCO;
        return objetivo;
    }

    public static int[] parsearEstado(String texto) {
        if (texto == null || texto.trim().isEmpty()) {
            throw new IllegalArgumentException("Debes ingresar 25 numeros (0-24).");
        }
        String[] tokens = texto.trim().split("[,\\s]+");
        if (tokens.length != TOTAL_CELDAS) {
            throw new IllegalArgumentException("Se esperaban 25 numeros y se recibieron " + tokens.length + ".");
        }

        int[] estado = new int[TOTAL_CELDAS];
        boolean[] vistos = new boolean[TOTAL_CELDAS];

        for (int i = 0; i < tokens.length; i++) {
            int valor;
            try {
                valor = Integer.parseInt(tokens[i]);
            } catch (NumberFormatException ex) {
                throw new IllegalArgumentException("Valor invalido: " + tokens[i]);
            }

            if (valor < 0 || valor >= TOTAL_CELDAS) {
                throw new IllegalArgumentException("Los valores deben estar entre 0 y 24.");
            }
            if (vistos[valor]) {
                throw new IllegalArgumentException("Valor repetido: " + valor);
            }

            vistos[valor] = true;
            estado[i] = valor;
        }

        return estado;
    }

    public static Nodo generarAleatorioResoluble(Random random, int movimientosDesdeMeta) {
        if (movimientosDesdeMeta < 0) {
            movimientosDesdeMeta = 0;
        }

        int[] estado = objetivo();
        int blanco = TOTAL_CELDAS - 1;
        int blancoAnterior = -1;

        for (int i = 0; i < movimientosDesdeMeta; i++) {
            List<Integer> vecinos = vecinosDelBlanco(blanco);
            if (vecinos.size() > 1 && blancoAnterior != -1) {
                vecinos.remove(Integer.valueOf(blancoAnterior));
            }

            int siguiente = vecinos.get(random.nextInt(vecinos.size()));
            intercambiar(estado, blanco, siguiente);
            blancoAnterior = blanco;
            blanco = siguiente;
        }

        return new Nodo(estado);
    }

    public List<Nodo> obtenerSucesores() {
        List<Nodo> sucesores = new ArrayList<>();
        int fila = indiceBlanco / LADO;
        int col = indiceBlanco % LADO;

        if (fila > 0) {
            sucesores.add(crearHijo(indiceBlanco - LADO, 'U'));
        }
        if (fila < LADO - 1) {
            sucesores.add(crearHijo(indiceBlanco + LADO, 'D'));
        }
        if (col > 0) {
            sucesores.add(crearHijo(indiceBlanco - 1, 'L'));
        }
        if (col < LADO - 1) {
            sucesores.add(crearHijo(indiceBlanco + 1, 'R'));
        }

        return sucesores;
    }

    public int calcularHeuristica(TipoHeuristica tipo) {
        int manhattan = calcularManhattan();
        if (tipo == TipoHeuristica.MANHATTAN) {
            return manhattan;
        }
        return manhattan + (2 * calcularConflictosLineales());
    }

    public boolean esObjetivo() {
        for (int i = 0; i < TOTAL_CELDAS - 1; i++) {
            if (estado[i] != i + 1) {
                return false;
            }
        }
        return estado[TOTAL_CELDAS - 1] == BLANCO;
    }

    public boolean esResoluble() {
        int inversiones = contarInversiones();
        if (LADO % 2 == 1) {
            return (inversiones % 2) == 0;
        }

        int filaBlancoDesdeAbajo = LADO - (indiceBlanco / LADO);
        if ((filaBlancoDesdeAbajo % 2) == 0) {
            return (inversiones % 2) == 1;
        }
        return (inversiones % 2) == 0;
    }

    public int[] copiarEstado() {
        return Arrays.copyOf(estado, estado.length);
    }

    public String getClave() {
        if (clave == null) {
            StringBuilder sb = new StringBuilder(TOTAL_CELDAS * 3);
            for (int i = 0; i < estado.length; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(estado[i]);
            }
            clave = sb.toString();
        }
        return clave;
    }

    public Nodo getPadre() {
        return padre;
    }

    public void setPadre(Nodo padre) {
        this.padre = padre;
    }

    public int getCosto() {
        return costo;
    }

    public void setCosto(int costo) {
        this.costo = costo;
    }

    public char getMovimiento() {
        return movimiento;
    }

    public void setMovimiento(char movimiento) {
        this.movimiento = movimiento;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < estado.length; i++) {
            int valor = estado[i];
            if (valor == BLANCO) {
                sb.append(" _ ");
            } else if (valor < 10) {
                sb.append(' ').append(valor).append(' ');
            } else {
                sb.append(valor).append(' ');
            }
            if ((i + 1) % LADO == 0) {
                sb.append('\n');
            }
        }
        return sb.toString();
    }

    private Nodo crearHijo(int nuevoIndiceBlanco, char mov) {
        int[] nuevoEstado = Arrays.copyOf(estado, estado.length);
        intercambiar(nuevoEstado, indiceBlanco, nuevoIndiceBlanco);
        Nodo hijo = new Nodo(nuevoEstado);
        hijo.setMovimiento(mov);
        return hijo;
    }

    private int calcularManhattan() {
        int total = 0;
        for (int i = 0; i < estado.length; i++) {
            int ficha = estado[i];
            if (ficha == BLANCO) {
                continue;
            }
            int filaActual = i / LADO;
            int colActual = i % LADO;
            int filaMeta = filaObjetivo(ficha);
            int colMeta = columnaObjetivo(ficha);
            total += Math.abs(filaActual - filaMeta) + Math.abs(colActual - colMeta);
        }
        return total;
    }

    private int calcularConflictosLineales() {
        int conflictos = 0;

        for (int fila = 0; fila < LADO; fila++) {
            for (int c1 = 0; c1 < LADO; c1++) {
                int ficha1 = estado[fila * LADO + c1];
                if (ficha1 == BLANCO || filaObjetivo(ficha1) != fila) {
                    continue;
                }
                for (int c2 = c1 + 1; c2 < LADO; c2++) {
                    int ficha2 = estado[fila * LADO + c2];
                    if (ficha2 == BLANCO || filaObjetivo(ficha2) != fila) {
                        continue;
                    }
                    if (columnaObjetivo(ficha1) > columnaObjetivo(ficha2)) {
                        conflictos++;
                    }
                }
            }
        }

        for (int col = 0; col < LADO; col++) {
            for (int f1 = 0; f1 < LADO; f1++) {
                int ficha1 = estado[f1 * LADO + col];
                if (ficha1 == BLANCO || columnaObjetivo(ficha1) != col) {
                    continue;
                }
                for (int f2 = f1 + 1; f2 < LADO; f2++) {
                    int ficha2 = estado[f2 * LADO + col];
                    if (ficha2 == BLANCO || columnaObjetivo(ficha2) != col) {
                        continue;
                    }
                    if (filaObjetivo(ficha1) > filaObjetivo(ficha2)) {
                        conflictos++;
                    }
                }
            }
        }

        return conflictos;
    }

    private int contarInversiones() {
        int inversiones = 0;
        for (int i = 0; i < estado.length; i++) {
            int a = estado[i];
            if (a == BLANCO) {
                continue;
            }
            for (int j = i + 1; j < estado.length; j++) {
                int b = estado[j];
                if (b == BLANCO) {
                    continue;
                }
                if (a > b) {
                    inversiones++;
                }
            }
        }
        return inversiones;
    }

    private static int buscarIndiceBlanco(int[] estado) {
        for (int i = 0; i < estado.length; i++) {
            if (estado[i] == BLANCO) {
                return i;
            }
        }
        throw new IllegalArgumentException("El estado no contiene el espacio en blanco (0).");
    }

    private static List<Integer> vecinosDelBlanco(int indiceBlanco) {
        List<Integer> vecinos = new ArrayList<>(4);
        int fila = indiceBlanco / LADO;
        int col = indiceBlanco % LADO;

        if (fila > 0) {
            vecinos.add(indiceBlanco - LADO);
        }
        if (fila < LADO - 1) {
            vecinos.add(indiceBlanco + LADO);
        }
        if (col > 0) {
            vecinos.add(indiceBlanco - 1);
        }
        if (col < LADO - 1) {
            vecinos.add(indiceBlanco + 1);
        }

        return vecinos;
    }

    private static int filaObjetivo(int ficha) {
        if (ficha == BLANCO) {
            return LADO - 1;
        }
        return (ficha - 1) / LADO;
    }

    private static int columnaObjetivo(int ficha) {
        if (ficha == BLANCO) {
            return LADO - 1;
        }
        return (ficha - 1) % LADO;
    }

    private static void intercambiar(int[] arreglo, int i, int j) {
        int tmp = arreglo[i];
        arreglo[i] = arreglo[j];
        arreglo[j] = tmp;
    }
}
