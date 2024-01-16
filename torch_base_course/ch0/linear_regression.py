import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

lr = 0.05

x = torch.rand(20, 1) * 10
# print(x[:4])
y = 2*x + (5 + torch.randn(20, 1))

w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)
    loss = (0.5 * (y - y_pred) ** 2).mean()

    loss.backward()

    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    b.grad.zero_()
    w.grad.zero_()

    # 绘图，每隔 20 次重新绘制直线
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        # 如果 MSE 小于 1，则停止训练
        if loss.data.numpy() < 1:
            break
