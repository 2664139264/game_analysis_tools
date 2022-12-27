import numpy as np
from copy import deepcopy
from typing import Callable, Tuple, Dict

class NFG:
    
    def __init__(self, utilities:np.ndarray, decimals:int=2):
        
        self.decimals = decimals
        
        self._n_players = utilities.shape[0]
        
        # 第一维大小为玩家个数，其余各维大小均为该玩家策略集合大小
        self._utilities = deepcopy(utilities)
        
        shape = self.utilities.shape[2:] + (1,)
        
        # 某维度及之后维度下一片的大小
        self._block_size = np.multiply.accumulate(shape[::-1])[::-1]
        
        # 效用函数张量的大小
        self._utilities_size = self._block_size[0] * self.utilities.shape[1]
        self._utilities_shape = self.utilities[0].shape
        
        # 全1的列向量
        self._ones_vertical = np.ones(shape=(self._utilities_size, 1), dtype=np.int32)
        self._ones_horizontal = self._ones_vertical.T  
        
        self._e_u = np.eye(self._utilities_size)
        
        # 计算所有需要用到的矩阵
        self._precalculate_matrices()      
        
    @property
    def n_players(self) -> int:
        return self._n_players
    
    @property
    def utilities(self) -> np.ndarray:
        return self._utilities
    
    # 将策略组合映射为编号
    def _strategy_profile_to_index(self, strategy_profile:tuple) -> int:
        return int(np.array(strategy_profile) @ self._block_size)
                                                    
    # 将编号映射回策略组合
    def _index_to_strategy_profile(self, index:int) -> tuple:
        return tuple(
            index // self._block_size[i] if i == 0 
            else (index % self._block_size[i-1]) // self._block_size[i] 
            for i in range(self.n_players)
        )
    
    # 选择比较矩阵
    def _select_comp_matrix(self, comp:Callable=None, comp_extra_param:int=None) -> np.ndarray:
        return (
            self._comparable_matrix if comp == self._is_comparable_strategy_profile 
            else self._m_comparable_matrices[comp_extra_param] if comp == self._is_m_comparable_strategy_profile
            else self._e_u
        )
    
    # 选择梯度算子伪逆矩阵
    def _select_flatten_grad_pseudo_inv_matrix(self, comp:Callable=None, comp_extra_param:int=None) -> np.ndarray:
        return (
            self._flatten_grad_operator_pseudo_inv if comp == self._is_comparable_strategy_profile
            else self._flatten_m_grad_operators_pseudo_invs[comp_extra_param] if comp == self._is_m_comparable_strategy_profile
            else self._flatten_grad_operator_pseudo_inv_without_comparability_check
        )
    
    # 在可比较的策略组合之间计算离散梯度
    # potential的维度为: (策略组合数,)
    def grad(self, potential:np.ndarray, comp:Callable=None, comp_extra_param:int=None) -> np.ndarray:
        comp_matrix = self._select_comp_matrix(comp, comp_extra_param)
        
        return (
            self._ones_vertical @ potential.reshape(self._ones_horizontal.shape) 
            - potential.reshape(self._ones_vertical.shape) @ self._ones_horizontal
        ) * comp_matrix
    
    # grad算子的伪逆
    # flow的维度为: (策略组合数,策略组合数)
    def grad_pseudo_inv(self, flow:np.ndarray, comp:Callable=None, comp_extra_param:int=None) -> np.ndarray:
        grad_pseudo_inv_matrix = self._select_flatten_grad_pseudo_inv_matrix(comp, comp_extra_param)
        return grad_pseudo_inv_matrix @ flow.flatten()
        
    # 梯度的伴随算子：散度
    # flow的维度：(策略组合数,策略组合数)
    def div(self, flow:np.ndarray, comp:Callable=None, comp_extra_param:int=None) -> np.ndarray:
        comp_matrix = self._select_comp_matrix(comp, comp_extra_param)
        return (flow.T * comp_matrix.T).sum(axis=0) - (flow * comp_matrix).sum(axis=1)
    
    # 投影算子，向non-strategic的正交补空间投影
    def pi(self, potential:np.ndarray, comp:Callable=None, comp_extra_param:int=None) -> np.ndarray:
        return self.grad_pseudo_inv(self.grad(potential, comp, comp_extra_param), comp, comp_extra_param)
    
    # 提前计算需要用到的矩阵
    def _precalculate_matrices(self) -> None:
        
        # 计算可比较性01矩阵
        self._m_comparable_matrices = [
            np.fromfunction(
                np.vectorize(
                    lambda p1, p2: self._is_m_comparable_strategy_profile(
                        self._index_to_strategy_profile(p1), self._index_to_strategy_profile(p2), m
                    ), signature='(),()->()'
                ), shape=(self._utilities_size, self._utilities_size), dtype=np.int32
            ) for m in range(self.n_players)
        ]
        
        self._comparable_matrix = np.sum(self._m_comparable_matrices, axis=0)
        
        # 构造梯度算子矩阵（将像的矩阵空间展平为向量空间）
        
        blocks = np.fromfunction(
            np.vectorize(
                lambda i: self._e_u - np.repeat(self._e_u[i].reshape(1, -1), self._utilities_size, axis=0),
                signature=f'()->({self._utilities_size},{self._utilities_size})'
            ), shape=(self._utilities_size,), dtype=np.int32
        )
        
        self._flatten_grad_operator = np.concatenate(blocks, axis=0) 
        
        self._flatten_grad_operator_pseudo_inv_without_comparability_check = np.linalg.pinv(self._flatten_grad_operator)
        
        self._flatten_grad_operator *= self._comparable_matrix.flatten().reshape(-1, 1)
        
        self._flatten_m_grad_operators = [
            self._flatten_grad_operator * self._m_comparable_matrices[i].flatten().reshape(-1, 1)
            for i in range(self.n_players)
        ]
        
        # 梯度算子求伪逆
        self._flatten_m_grad_operators_pseudo_invs = [np.linalg.pinv(op) for op in self._flatten_m_grad_operators]
        self._flatten_grad_operator_pseudo_inv = np.linalg.pinv(self._flatten_grad_operator)
        
        # 计算势向量
        self._potential = sum(self.grad(self.utilities[i].flatten(), self._is_m_comparable_strategy_profile, i) for i in range(self.n_players))
        self._potential = self.grad_pseudo_inv(self._potential, self._is_comparable_strategy_profile)
        
    
    # 对于玩家m来说可比的策略组合
    def _is_m_comparable_strategy_profile(self, profile_1:Tuple, profile_2:Tuple, m:int) -> bool:
        profile_diff = np.array(profile_1) - np.array(profile_2)
        profile_diff[m] = 0
        return (profile_diff == 0).all()
    
    # 对某个玩家来说可比的策略组合
    def _is_comparable_strategy_profile(self, profile_1:Tuple, profile_2:Tuple) -> bool:
        profile_diff = np.array(profile_1) - np.array(profile_2)
        max_index = profile_diff.argmax()
        min_index = profile_diff.argmin()
        
        if profile_diff[max_index] != 0 and profile_diff[min_index] != 0:
            return False
        
        if profile_diff[max_index] != 0:
            profile_diff[max_index] = 0
            return (profile_diff == 0).all()
        
        if profile_diff[min_index] != 0:
            profile_diff[min_index] = 0
            return (profile_diff == 0).all()
        
        return True 

    @property
    def potential(self) -> np.ndarray:
        return self._potential
    
    def decompose(self) -> Dict[str, np.ndarray]:
        potential_components = [
            self.pi(self.potential, self._is_m_comparable_strategy_profile, i).reshape(self._utilities_shape)
            for i in range(self.n_players)
        ]
        harmonic_components = [
            self.pi(self.utilities[i].flatten(), self._is_m_comparable_strategy_profile, i).reshape(self._utilities_shape) - potential_components[i]
            for i in range(self.n_players)
        ]
        nonstrategic_components = [
            self.utilities[i] - potential_components[i] - harmonic_components[i]
            for i in range(self.n_players)
        ]
        
        return {
            'potential_components' : np.around(potential_components, self.decimals), 
            'harmonic_components' : np.around(harmonic_components, self.decimals), 
            'nonstrategic_components' : np.around(nonstrategic_components, self.decimals)
        }

if __name__ == '__main__':
    
    # x=1, y=2, z=4
    
    utilities = np.array(
        [[[0, -3, 6],
          [3, 0, -12],
          [-6, 12, 0]],
         [[0, 3, -6],
          [-3, 0, 12],
          [6, -12, 0]]],
        dtype=np.float32
    )
    
    g_rpl = NFG(utilities)
    
    print(g_rpl.decompose())
    
