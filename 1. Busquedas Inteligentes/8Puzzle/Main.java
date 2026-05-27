package puzzle8;

public class Main {
    public static void main(String[] args) {

        Estado estadoInicial = new Estado("1238 4765");
        Buscador8Puzzle solver = new Buscador8Puzzle(estadoInicial);

        Estado resultado = solver.bfs("1284376 5");

        if (resultado == null) {
            System.out.println("No se encontro solucion para el estado objetivo.");
            return;
        }

        System.out.println("Estado objetivo alcanzado: " + resultado.estado);
        System.out.println("Profundidad final: " + resultado.profundidad);
        System.out.println();
        resultado.mostrarRuta();
    }
}
