package puzzle8;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Buscador8Puzzle {

    public Estado estadoInicial;

    public Buscador8Puzzle(Estado estadoInicial) {
        this.estadoInicial = estadoInicial;
    }

    public Estado bfs(String objetivo) {
        if (estadoInicial == null) return null;

        objetivo = objetivo.replace('0', ' ');

        HashSet<String> visitados = new HashSet<>();
        Queue<Estado> pendientes = new LinkedList<>();

        pendientes.add(estadoInicial);
        visitados.add(estadoInicial.estado);

        while (!pendientes.isEmpty()) {
            Estado actual = pendientes.poll();

            if (actual.estado.equals(objetivo)) {
                return actual;
            }

            List<Estado> vecinos = actual.expandir();
            for (Estado candidato : vecinos) {
                if (!visitados.contains(candidato.estado)) {
                    visitados.add(candidato.estado);
                    pendientes.add(candidato);
                }
            }
        }

        return null;
    }
}
