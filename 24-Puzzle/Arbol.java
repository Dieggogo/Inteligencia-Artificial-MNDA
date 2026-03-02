import java.util.HashSet;

public class Arbol {
    private final Nodo raiz;

    private long nodosExpandidos;
    private long tiempoNs;
    private int longitudSolucion;

    private static class ResultadoDFS {
        private final Nodo solucion;
        private final int siguienteUmbral;

        ResultadoDFS(Nodo solucion, int siguienteUmbral) {
            this.solucion = solucion;
            this.siguienteUmbral = siguienteUmbral;
        }
    }

    public Arbol(Nodo raiz) {
        this.raiz = raiz;
        this.nodosExpandidos = 0L;
        this.tiempoNs = 0L;
        this.longitudSolucion = -1;
    }

    public Nodo resolverIDAStar(Nodo.TipoHeuristica tipoHeuristica) {
        nodosExpandidos = 0L;
        longitudSolucion = -1;
        tiempoNs = 0L;

        if (raiz == null || !raiz.esResoluble()) {
            return null;
        }

        raiz.setPadre(null);
        raiz.setCosto(0);
        raiz.setMovimiento('S');

        int umbral = raiz.calcularHeuristica(tipoHeuristica);
        long inicio = System.nanoTime();

        while (true) {
            HashSet<String> enRuta = new HashSet<>();
            enRuta.add(raiz.getClave());

            ResultadoDFS resultado = dfsLimitado(raiz, 0, umbral, tipoHeuristica, enRuta);

            if (resultado.solucion != null) {
                tiempoNs = System.nanoTime() - inicio;
                longitudSolucion = resultado.solucion.getCosto();
                return resultado.solucion;
            }

            if (resultado.siguienteUmbral == Integer.MAX_VALUE) {
                tiempoNs = System.nanoTime() - inicio;
                return null;
            }

            umbral = resultado.siguienteUmbral;
        }
    }

    public long getNodosExpandidos() {
        return nodosExpandidos;
    }

    public long getTiempoNs() {
        return tiempoNs;
    }

    public int getLongitudSolucion() {
        return longitudSolucion;
    }

    private ResultadoDFS dfsLimitado(
        Nodo actual,
        int g,
        int umbral,
        Nodo.TipoHeuristica tipoHeuristica,
        HashSet<String> enRuta
    ) {
        int h = actual.calcularHeuristica(tipoHeuristica);
        int f = g + h;
        if (f > umbral) {
            return new ResultadoDFS(null, f);
        }

        if (actual.esObjetivo()) {
            return new ResultadoDFS(actual, umbral);
        }

        nodosExpandidos++;
        int minimoExcedido = Integer.MAX_VALUE;

        for (Nodo hijo : actual.obtenerSucesores()) {
            String claveHijo = hijo.getClave();
            if (enRuta.contains(claveHijo)) {
                continue;
            }

            hijo.setPadre(actual);
            hijo.setCosto(g + 1);

            enRuta.add(claveHijo);
            ResultadoDFS resultado = dfsLimitado(hijo, g + 1, umbral, tipoHeuristica, enRuta);

            if (resultado.solucion != null) {
                return resultado;
            }

            if (resultado.siguienteUmbral < minimoExcedido) {
                minimoExcedido = resultado.siguienteUmbral;
            }

            enRuta.remove(claveHijo);
        }

        return new ResultadoDFS(null, minimoExcedido);
    }
}
