import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Set Streamlit layout to wide
st.set_page_config(
    page_title="Gradient Descent Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for input parameters
st.sidebar.header("Gradient Descent Parameters")
learning_rate = st.sidebar.slider(
    "Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01
)
max_iterations = st.sidebar.number_input(
    "Max Iterations", min_value=10, max_value=500, value=100
)
start_x = st.sidebar.number_input("Start X", min_value=-5.0, max_value=5.0, value=1.5)
start_y = st.sidebar.number_input("Start Y", min_value=-5.0, max_value=5.0, value=1.5)

# Main section
st.title("Gradient Descent Simulation")
st.write("Adjust the parameters in the sidebar and start the simulation.")


# Gradient descent
def gradient_descent_step(x, y, lr):
    grad_x = 2 * x
    grad_y = 2 * y
    x -= lr * grad_x
    y -= lr * grad_y
    return x, y, x**2 + y**2


if st.button("Start Gradient Descent"):
    x, y = start_x, start_y  # Starting point from user input
    trajectory = [(x, y)]
    losses = []

    for i in range(int(max_iterations)):
        x, y, loss = gradient_descent_step(x, y, learning_rate)
        trajectory.append((x, y))
        losses.append(loss)

    st.write("Simulation completed!")
    st.write(f"Final position: x = {x}, y = {y}, Loss = {loss}")
    st.write(f"Total iterations: {len(losses)}")

    # Use two columns for displaying plots
    col1, col2 = st.columns(2)

    # Loss Curve Plot in col1
    with col1:
        st.subheader("Loss Curve")
        st.write("")  # Add spacing for alignment
        fig2d = go.Figure()
        fig2d.add_trace(go.Scatter(y=losses, mode="lines", name="Loss"))
        fig2d.update_layout(
            xaxis_title="Iterations",
            yaxis_title="Loss",
            title="Loss Curve",
        )
        st.plotly_chart(fig2d)

    # 3D Surface Plot in col2
    with col2:
        st.subheader("3D Surface")
        X = np.linspace(-2, 2, 100)
        Y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(X, Y)
        Z = X**2 + Y**2

        # Convert trajectory to arrays for plotting
        trajectory = np.array(trajectory)
        traj_x = trajectory[:, 0]
        traj_y = trajectory[:, 1]
        traj_z = traj_x**2 + traj_y**2

        # Create 3D Surface with Plotly
        fig3d = go.Figure(
            data=[
                go.Surface(
                    z=Z, x=X, y=Y, colorscale="Viridis", opacity=0.8, showscale=False
                ),
                go.Scatter3d(
                    x=traj_x,
                    y=traj_y,
                    z=traj_z,
                    mode="lines+markers",
                    marker=dict(size=4, color="red"),
                    line=dict(color="red", width=2),
                ),
            ]
        )
        fig3d.update_layout(
            title="Interactive 3D Surface",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
        )
        st.plotly_chart(fig3d)
