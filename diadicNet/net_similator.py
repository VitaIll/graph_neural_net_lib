import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

class NetSimulator:
    def __init__(self, num_employees, connection_prob_params, productivity_boost):
        self.num_employees = num_employees
        self.connection_prob_params = connection_prob_params 
        self.productivity_boost = productivity_boost

        self.G = nx.Graph()
        self.G.add_nodes_from(range(num_employees))

        self.intrinsic_productivity = np.random.uniform(0, 1, num_employees)

        alpha, beta = self.connection_prob_params
        self.connection_probabilities = st.beta.rvs(alpha, beta, size=self.num_employees)
        self.beta_moments = st.beta(alpha, beta).stats(moments='mvs')

        for i in range(num_employees):
            for j in range(i + 1, num_employees):
                if np.random.rand() < self.connection_probabilities[i] * self.connection_probabilities[j]:
                    self.G.add_edge(i, j)
       
        adjacency_matrix_sparse = nx.adjacency_matrix(self.G)

        # If you need a dense NumPy array representation
        self.adj_mat = adjacency_matrix_sparse.toarray()


    def calculate_total_productivity(self):
        total_productivity = self.intrinsic_productivity.copy()
        for node in self.G.nodes:
            neighbors_productivity = sum(self.intrinsic_productivity[neighbor] for neighbor in self.G.neighbors(node))
            total_productivity[node] += self.productivity_boost * neighbors_productivity
        return total_productivity

    def visualize_network(self):
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
        plt.title("Employee Network")
        plt.show()

    def visualize_distributions(self):
        data = pd.DataFrame({
            'Intrinsic Productivity': self.intrinsic_productivity,
            'Total Productivity': self.calculate_total_productivity(),
            'Connection Probability': self.connection_probabilities
        })

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sns.histplot(data['Intrinsic Productivity'], kde=True, ax=axes[0], color='blue', label='Empirical')
        axes[0].set_title('Intrinsic Productivity Distribution')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='Population Mean (Uniform)')
        axes[0].legend()

        sns.histplot(data['Total Productivity'], kde=True, ax=axes[1], color='green')
        axes[1].set_title('Total Productivity Distribution')

        sns.histplot(data['Connection Probability'], kde=True, ax=axes[2], color='purple')
        axes[2].set_title('Connection Probability Distribution')

        plt.tight_layout()
        plt.show()

    def run_simulation(self, visulals: bool = True):

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_employees))

        for i in range(self.num_employees):
            for j in range(i + 1, self.num_employees):
                if np.random.rand() < self.connection_probabilities[i] * self.connection_probabilities[j]:
                    self.G.add_edge(i, j)
       
        adjacency_matrix_sparse = nx.adjacency_matrix(self.G)

        # If you need a dense NumPy array representation
        self.adj_mat = adjacency_matrix_sparse.toarray()
       
        if visulals:
            self.visualize_network()
            self.visualize_distributions()
        else:
            pass

        self.result_dict = {
            'Employee ID': range(self.num_employees),
            'Intrinsic Productivity': self.intrinsic_productivity,
            'Total Productivity': self.calculate_total_productivity(),
            'Connection Probability': self.connection_probabilities,
            'Total Product': sum(self.calculate_total_productivity()),
            'Adjaceny Matrix': self.adj_mat,
            'Beta moments': self.beta_moments
        }
        
        results = pd.DataFrame({
            'Employee ID': range(self.num_employees),
            'Intrinsic Productivity': self.intrinsic_productivity,
            'Total Productivity': self.calculate_total_productivity(),
            'Connection Probability': self.connection_probabilities
        })
        return results