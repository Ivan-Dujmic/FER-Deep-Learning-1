import torch
import torch.optim as optim
import matplotlib.pyplot as plt

class PTLinreg:
    def train(self, X, Y_, lr=0.1, steps=1000, print_step=50, manual_grad=False):
        """
        Parameters:
            X - data
            Y_ - true classes
            lr - learning rate
            steps - iterations
            print_step - print loss every # of steps
            manual_grad - calculate grad without pytorch and print it
        """
        a = torch.randn(1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)

        optimizer = optim.SGD([a, b], lr=lr)

        for i in range(steps):
            Y = a * X + b
            diff = Y_ - Y
            loss = torch.sum(diff ** 2)
            loss.backward()
            if print_step != 0 and i % print_step == 0:
                if manual_grad:
                    grad_a = -2 * torch.sum(X * (Y_ - (a * X + b)))
                    grad_b = -2 * torch.sum(Y_ - (a * X + b))
                    print(f"step: {i}, loss: {loss:.4f}, a: {a.item():.4f}, b: {b.item():.4f}, grad_a: {a.grad.item():.4f}, man_grad_a: {grad_a.item():.4f}, grad_b: {b.grad.item():.4f}, man_grad_b: {grad_b.item():.4f}")
                else:
                    print(f"step: {i}, loss: {loss:.4f}, a: {a.item():.4f}, b: {b.item():.4f}, grad_a: {a.grad.item():.4f}, grad_b: {b.grad.item():.4f}")
            optimizer.step()
            optimizer.zero_grad()

        if print_step != 0:
            Y = a * X + b
            diff = Y_ - Y
            loss = torch.sum(diff ** 2)
            loss.backward()
            if manual_grad:
                grad_a = -2 * torch.sum(X * (Y_ - (a * X + b)))
                grad_b = -2 * torch.sum(Y_ - (a * X + b))
                print(f"FINAL: loss: {loss:.4f}, a: {a.item():.4f}, b: {b.item():.4f}, grad_a: {a.grad.item():.4f}, man_grad_a: {grad_a.item():.4f}, grad_b: {b.grad.item():.4f}, man_grad_b: {grad_b.item():.4f}")
            else:
                print(f"FINAL: loss: {loss:.4f}, a: {a.item():.4f}, b: {b.item():.4f}, grad_a: {a.grad.item():.4f}, grad_b: {b.grad.item():.4f}")
        
        self.a = a
        self.b = b
            
    def predict(self, X):
        """
        Parameters:
            X - data

        Returns: predictions
        """
        return (self.a * X + self.b).detach().numpy()

if __name__=="__main__":
    a = -0.25
    b = 1.5
    spread = 10
    X = torch.rand(50) * 10
    Y_ = a * (X + (torch.rand(50) - 0.5) * spread) + b

    model = PTLinreg()
    model.train(X, Y_, 0.0001, manual_grad=True)

    x_line = torch.linspace(X.min().item(), X.max().item(), 100)
    y_line = (model.a * x_line + model.b).detach().numpy()
    plt.scatter(X.numpy(), Y_)
    plt.plot(x_line.numpy(), y_line, color='red')
    plt.show()