import java.util.LinkedList;
import java.util.Random;
import java.util.Scanner;

public class App {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        Random random = new Random();

        System.out.println("=== 24-Puzzle (5x5) con IDA* ===");
        System.out.println("Estado objetivo:");
        System.out.print(new Nodo(Nodo.objetivo()));

        Nodo inicial = solicitarEstadoInicial(scanner, random);
        if (inicial == null) {
            scanner.close();
            return;
        }

        if (!inicial.esResoluble()) {
            System.out.println("El estado ingresado no es resoluble para el 24-puzzle.");
            scanner.close();
            return;
        }

        int[] estadoBase = inicial.copiarEstado();

        boolean continuar = true;
        while (continuar) {
            System.out.println();
            System.out.println("Selecciona una opcion:");
            System.out.println("1. Resolver con Manhattan");
            System.out.println("2. Resolver con Conflicto Lineal");
            System.out.println("3. Comparar ambas heuristicas");
            System.out.println("4. Salir");
            int opcion = leerEntero(scanner);

            switch (opcion) {
                case 1:
                    ejecutarYMostrar("Manhattan", Nodo.TipoHeuristica.MANHATTAN, estadoBase);
                    break;
                case 2:
                    ejecutarYMostrar("Conflicto Lineal", Nodo.TipoHeuristica.CONFLICTO_LINEAL, estadoBase);
                    break;
                case 3:
                    TablaComparativa.Resultado resultadoManhattan =
                        ejecutarYMostrar("Manhattan", Nodo.TipoHeuristica.MANHATTAN, estadoBase);
                    TablaComparativa.Resultado resultadoConflicto =
                        ejecutarYMostrar("Conflicto Lineal", Nodo.TipoHeuristica.CONFLICTO_LINEAL, estadoBase);
                    TablaComparativa.imprimirTabla(resultadoManhattan, resultadoConflicto);
                    break;
                case 4:
                    continuar = false;
                    break;
                default:
                    System.out.println("Opcion invalida.");
                    break;
            }
        }

        scanner.close();
    }

    private static Nodo solicitarEstadoInicial(Scanner scanner, Random random) {
        while (true) {
            System.out.println();
            System.out.println("Estado inicial:");
            System.out.println("1. Ingresar estado manual");
            System.out.println("2. Generar estado aleatorio resoluble");
            int opcion = leerEntero(scanner);

            if (opcion == 1) {
                System.out.println("Ingresa 25 valores (0-24), separados por espacios. Usa 0 como espacio en blanco.");
                String linea = scanner.nextLine().trim();
                try {
                    Nodo nodo = new Nodo(Nodo.parsearEstado(linea));
                    System.out.println("Estado inicial:");
                    System.out.print(nodo);
                    return nodo;
                } catch (IllegalArgumentException ex) {
                    System.out.println("Error: " + ex.getMessage());
                }
            } else if (opcion == 2) {
                int movimientos = 15 + random.nextInt(26); // rango [15, 40]
                Nodo nodo = Nodo.generarAleatorioResoluble(random, movimientos);
                System.out.println("Estado inicial generado (mezclado con " + movimientos + " movimientos):");
                System.out.print(nodo);
                return nodo;
            } else {
                System.out.println("Opcion invalida.");
            }
        }
    }

    private static TablaComparativa.Resultado ejecutarYMostrar(
        String nombre,
        Nodo.TipoHeuristica tipoHeuristica,
        int[] estadoBase
    ) {
        Nodo raiz = new Nodo(estadoBase);
        Arbol arbol = new Arbol(raiz);

        System.out.println();
        System.out.println("Resolviendo con: " + nombre);

        Nodo solucion = arbol.resolverIDAStar(tipoHeuristica);
        TablaComparativa.Resultado resultado = TablaComparativa.desdeArbol(nombre, arbol);

        if (solucion == null) {
            System.out.println("No se encontro solucion.");
        } else {
            imprimirCamino(solucion);
        }

        System.out.printf(
            "Resumen -> Nodos: %d | Tiempo(s): %.6f | Longitud: %d%n",
            resultado.getNodosExpandidos(),
            resultado.getTiempoSegundos(),
            resultado.getLongitudSolucion()
        );

        return resultado;
    }

    private static void imprimirCamino(Nodo nodoSolucion) {
        LinkedList<Nodo> camino = new LinkedList<>();
        Nodo actual = nodoSolucion;
        while (actual != null) {
            camino.addFirst(actual);
            actual = actual.getPadre();
        }

        StringBuilder secuencia = new StringBuilder();
        for (int i = 0; i < camino.size(); i++) {
            Nodo paso = camino.get(i);
            System.out.println("Paso " + i + ":");
            System.out.print(paso);
            if (i > 0) {
                if (secuencia.length() > 0) {
                    secuencia.append(' ');
                }
                secuencia.append(paso.getMovimiento());
            }
        }

        System.out.println("Secuencia de movimientos (U,D,L,R): " + secuencia);
        System.out.println("Movimientos totales: " + (camino.size() - 1));
    }

    private static int leerEntero(Scanner scanner) {
        while (true) {
            String linea = scanner.nextLine().trim();
            try {
                return Integer.parseInt(linea);
            } catch (NumberFormatException ex) {
                System.out.println("Ingresa un numero entero valido.");
            }
        }
    }
}
