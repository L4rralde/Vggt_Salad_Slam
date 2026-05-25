import torch

from .core import Algorithm


class GaussNewton(Algorithm):
    def optimize(self, n_iter: int) -> None:
        for i in range(n_iter):
            H, b = self.compute_h_and_b()
            d = torch.linalg.solve(H, -b).view(self.n, self.m) #Descent direction
            print(f"{i+1}. Loss: {self.loss()}. d.norm: {torch.linalg.norm(d)}")
            #Now we must compute 
            if torch.linalg.norm(d) < self.eps:
                break
            alpha = self.step_size(d)
            self.update_vertices(alpha * d)
        self.update_edges()


class LevenbergMarquardt(Algorithm):
    def optimize(self, n_iter: int) -> None:
        # Inicializamos lambda. Un valor pequeño confía en Gauss-Newton, 
        # uno grande se comporta como Descenso de Gradiente.
        lambd = 1e-3 
        
        for i in range(n_iter):
            H, b = self.compute_h_and_b()
            
            # La clave de LM: Añadimos lambd a la diagonal de H
            # H_lm = H + lambd * I
            I = torch.eye(H.size(0), device=H.device)
            
            # Guardamos el error actual para comparar después del paso
            current_loss = self.loss()
            
            # Resolvemos el sistema: (H + lambd*I)d = -b
            try:
                H_lm = H + lambd * I
                d = torch.linalg.solve(H_lm, -b).view(self.n, self.m)
            except RuntimeError:
                # Si la matriz es singular, aumentamos lambd y saltamos
                lambd *= 10
                continue

            if torch.linalg.norm(d) < self.eps:
                break

            # --- Estrategia de aceptación del paso ---
            # Guardamos estado actual por si hay que retroceder
            old_vertices = [v.copy() for v in self._vertices]
            
            alpha = self.step_size(d)
            self.update_vertices(alpha * d)
            new_loss = self.loss()

            if new_loss < current_loss:
                # ¡Éxito! El paso redujo el error. 
                # Nos movemos hacia Gauss-Newton (reducimos lambd)
                lambd /= 10
                print(f"{i+1}. Success - Loss: {new_loss:.6f}, Lambda: {lambd:.2e}")
            else:
                # El error aumentó. Rechazamos el paso y aumentamos lambd
                # (nos volvemos más conservadores/Descenso de Gradiente)
                self._vertices = old_vertices
                lambd *= 10
                print(f"{i+1}. Rejected - Increasing Lambda: {lambd:.2e}")

        self.update_edges()