import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

def generate_ramp(duration, dt, slope):
    t = 0.0
    times = []
    values = []
    while t <= duration:
        value = slope * t
        times.append(t)
        values.append(value)
        t += dt
    return times, values

def generate_sinusoid(duration, dt, amplitude, frequency):
    t = 0.0
    times = []
    values = []
    while t <= duration:
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        times.append(t)
        values.append(value)
        t += dt
    return times, values

def generate_circle(duration, dt, frequency, radius):    
    t = 0.0
    times = []
    x_values = []
    y_values = []
    while t <= duration:
        x = radius * math.cos(2 * math.pi * frequency * t)
        y = radius * math.sin(2 * math.pi * frequency * t)
        times.append(t)
        x_values.append(x)
        y_values.append(y)
        t += dt
    return x_values, y_values

def generate_ellipse(duration, dt, frequency, radius_x, radius_y):    
    t = 0.0
    times = []
    x_values = []
    y_values = []
    while t <= duration:
        x = radius_x * math.cos(2 * math.pi * frequency * t)
        y = radius_y * math.sin(2 * math.pi * frequency * t)
        times.append(t)
        x_values.append(x)
        y_values.append(y)
        t += dt
    return x_values, y_values

def plot_1d(times, values, title):
    plt.plot(times, values)
    plt.xlabel('Time [s]')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_2d(x_values, y_values, title, animate=False):
    fig, ax = plt.subplots()
    if not animate:
        ax.plot(x_values, y_values)
    else:
        ax.plot(x_values, y_values, alpha = 0.3)
        punto, = ax.plot([], [], 'ro', markersize=10)

        def update(frame):
            punto.set_data([x_values[frame]], [y_values[frame]])
            return punto,

        animation.FuncAnimation(fig, update, frames=len(x_values), interval=30, repeat=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True)
    ax.axis('equal')
    plt.show()

def main():
    print("Which trajectory do you want to see?")
    print("1 - Ramp")
    print("2 - Sinusoid")
    print("3 - Circle")
    print("4 - Ellipse")
    choice = input("Write the number of your choice:")

    if choice == "1":
        times, values = generate_ramp(duration=10.0, dt=0.1, slope=0.1)
        plot_1d(times, values, 'Ramp trajectory')
    elif choice == "2":
        times, values = generate_sinusoid(duration=10.0, dt=0.1, amplitude=2.0, frequency=0.5)
        plot_1d(times, values, 'Sinusoidal trajectory')
    elif choice == "3":
        x_values, y_values = generate_circle(duration=10.0, dt=0.1, radius=2.0, frequency=0.2)
        plot_2d(x_values, y_values, 'Circular trajectory')
    elif choice == "4":
        x_values, y_values = generate_ellipse(duration=10.0, dt=0.1, radius_x=2.0, radius_y=1.0, frequency=0.2)
        plot_2d(x_values, y_values, 'Elliptical trajectory', animate=True)
    else:
        print('Invalid choice.')

if __name__ == '__main__':
    main()

