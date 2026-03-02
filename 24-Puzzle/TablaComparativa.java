public class TablaComparativa {

    public static class Resultado {
        private final String nombre;
        private final long nodosExpandidos;
        private final double tiempoSegundos;
        private final int longitudSolucion;

        public Resultado(String nombre, long nodosExpandidos, double tiempoSegundos, int longitudSolucion) {
            this.nombre = nombre;
            this.nodosExpandidos = nodosExpandidos;
            this.tiempoSegundos = tiempoSegundos;
            this.longitudSolucion = longitudSolucion;
        }

        public String getNombre() {
            return nombre;
        }

        public long getNodosExpandidos() {
            return nodosExpandidos;
        }

        public double getTiempoSegundos() {
            return tiempoSegundos;
        }

        public int getLongitudSolucion() {
            return longitudSolucion;
        }
    }

    public static Resultado desdeArbol(String nombre, Arbol arbol) {
        double tiempoSegundos = arbol.getTiempoNs() / 1_000_000_000.0;
        return new Resultado(nombre, arbol.getNodosExpandidos(), tiempoSegundos, arbol.getLongitudSolucion());
    }

    public static void imprimirTabla(Resultado... resultados) {
        System.out.println("--------------------------------------------------------------------------------");
        System.out.println(" Heuristica               | Nodos expandidos |  Tiempo (s) | Longitud solucion");
        System.out.println("--------------------------------------------------------------------------------");

        for (Resultado resultado : resultados) {
            if (resultado == null) {
                continue;
            }
            String longitud = (resultado.getLongitudSolucion() >= 0)
                ? String.valueOf(resultado.getLongitudSolucion())
                : "Sin solucion";

            System.out.printf(
                " %-24s | %16d | %11.6f | %17s%n",
                resultado.getNombre(),
                resultado.getNodosExpandidos(),
                resultado.getTiempoSegundos(),
                longitud
            );
        }

        System.out.println("--------------------------------------------------------------------------------");
    }
}
