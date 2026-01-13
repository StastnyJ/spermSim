import numpy as np
from typing import Callable, List, Tuple, Dict
import json


class ActivationFunctions:   
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh(x: float) -> float:
        return np.tanh(x)
    
    @staticmethod
    def hill(x: float, n: float = 2.0, K: float = 0.5) -> float:
        x_n = abs(x) ** n
        K_n = K ** n
        result = x_n / (K_n + x_n)
        return result if x >= 0 else -result
    
    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)
    
    @staticmethod
    def get_function(name: str) -> Callable[[float], float]:
        functions = {
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'hill': ActivationFunctions.hill,
            'relu': ActivationFunctions.relu,
        }
        if name not in functions:
            raise ValueError(f"Unknown activation function: {name}")
        return functions[name]
    
class Gene:   
    def __init__(self, name: str, initial_value: float, bias: float,  activation: str, connections: List[Tuple[str, float]]):
        self.name = name
        self.value = initial_value
        self.bias = bias
        self.activation_func = ActivationFunctions.get_function(activation)
        self.connections: List[Tuple[str, float]] = connections
    
    def __repr__(self):
        return f"Gene({self.name}, value={self.value:.3f})"
    
class GRN:   
    def __init__(self, config_path: str, dt: float = 0.1):
        self.dt = dt
        self.config_path = config_path
        self.genes: Dict[str, Gene] = {}
        self.input_genes: List[str] = []
        self.internal_genes: List[str] = []
        self.output_genes: List[str] = []
        
        self._load_config(config_path)
        self._build_connection_matrix()
    
    def _load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        for gene_data in config['inputs']:
            gene = Gene(
                name=gene_data['name'],
                initial_value=gene_data.get('initial_value', 0.0),
                bias=gene_data.get('bias', 0.0),
                activation=gene_data.get('activation', 'sigmoid'),
                connections=gene_data.get('connections', [])
            )
            self.genes[gene.name] = gene
            self.input_genes.append(gene.name)
        
        for gene_data in config.get('internal_genes', []):
            gene = Gene(
                name=gene_data['name'],
                initial_value=gene_data['initial_value'],
                bias=gene_data['bias'],
                activation=gene_data['activation'],
                connections=gene_data['connections']
            )
            self.genes[gene.name] = gene
            self.internal_genes.append(gene.name)
        
        for gene_data in config['outputs']:
            gene = Gene(
                name=gene_data['name'],
                initial_value=gene_data.get('initial_value', 0.0),
                bias=gene_data.get('bias', 0.0),
                activation=gene_data.get('activation', 'sigmoid'),
                connections=gene_data.get('connections', [])
            )
            self.genes[gene.name] = gene
            self.output_genes.append(gene.name)
    
    def _build_connection_matrix(self):
        self.incoming_connections: Dict[str, List[Tuple[str, float]]] = {
            name: [] for name in self.genes.keys()
        }
        
        for source_name, gene in self.genes.items():
            for target_name, weight in gene.connections:
                if target_name in self.genes:
                    self.incoming_connections[target_name].append((source_name, weight))
    
    def set_input(self, input_values: Dict[str, float]):
        for name, value in input_values.items():
            if name in self.genes and name in self.input_genes:
                self.genes[name].value = np.clip(value, 0.0, 1.0)
    
    def update(self):
        new_values = {}
        
        genes_to_update = self.internal_genes + self.output_genes
        for gene_name in genes_to_update:
            gene = self.genes[gene_name]
            
            weighted_sum = 0.0
            for source_name, weight in self.incoming_connections[gene_name]:
                weighted_sum += weight * self.genes[source_name].value
            
            net_input = weighted_sum - gene.bias
            
            activation_output = gene.activation_func(net_input)
            
            new_value = gene.value + self.dt * (activation_output - gene.value)
            
            new_values[gene_name] = np.clip(new_value, 0.0, 1.0)
        
        for gene_name, value in new_values.items():
            self.genes[gene_name].value = value
    
    def get_output(self) -> Dict[str, float]:
        return {name: self.genes[name].value for name in self.output_genes}
    
    def get_all_values(self) -> Dict[str, float]:
        return {name: gene.value for name, gene in self.genes.items()}
    
    def reset(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        for gene_data in config['inputs'] + config.get('internal_genes', []) + config['outputs']:
            name = gene_data['name']
            if name in self.genes:
                self.genes[name].value = gene_data.get('initial_value', 0.0)
