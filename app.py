import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from itertools import combinations
import json
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="CRAFT Layout Optimizer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .improvement-card {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    .info-card {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏭 CRAFT Layout Optimizer")
st.markdown("**Computerized Relative Allocation of Facilities Technique** - Otimização de Layouts Industriais")

# ============================================================================
# FUNÇÕES DO ALGORITMO CRAFT
# ============================================================================

class CRAFTOptimizer:
    """Implementação do algoritmo CRAFT para otimização de layouts"""
    
    def __init__(self, departments, flows, distance_metric='manhattan'):
        """
        Inicializa o otimizador CRAFT
        
        Args:
            departments: dict com {id: {'name': str, 'area': float}}
            flows: dict com {(i, j): flow_value}
            distance_metric: 'manhattan' ou 'euclidean'
        """
        self.departments = departments
        self.flows = flows
        self.distance_metric = distance_metric
        self.history = []
        self.swap_history = []
        
    def calculate_distance(self, pos1, pos2):
        """Calcula distância entre dois centros de departamentos"""
        if self.distance_metric == 'manhattan':
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        else:  # euclidean
            return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_total_cost(self, layout):
        """
        Calcula o custo total de movimentação
        layout: dict com {dept_id: (x, y, width, height)}
        """
        total_cost = 0
        for (i, j), flow in self.flows.items():
            if i in layout and j in layout:
                x1, y1, w1, h1 = layout[i]
                x2, y2, w2, h2 = layout[j]
                
                # Centro de cada departamento
                center1 = (x1 + w1/2, y1 + h1/2)
                center2 = (x2 + w2/2, y2 + h2/2)
                
                distance = self.calculate_distance(center1, center2)
                total_cost += flow * distance
        
        return total_cost
    
    def generate_initial_layout(self, grid_width, grid_height, arrangement='random'):
        """
        Gera um layout inicial
        arrangement: 'random', 'ordered' ou 'by_area'
        """
        dept_ids = list(self.departments.keys())
        
        if arrangement == 'ordered':
            dept_ids.sort()
        elif arrangement == 'by_area':
            dept_ids.sort(key=lambda x: self.departments[x]['area'], reverse=True)
        else:
            np.random.shuffle(dept_ids)
        
        layout = {}
        x, y = 0, 0
        
        for dept_id in dept_ids:
            area = self.departments[dept_id]['area']
            width = min(int(np.sqrt(area)), grid_width)
            height = int(area / width) if width > 0 else 1
            
            if x + width > grid_width:
                x = 0
                y += height
            
            if y + height > grid_height:
                height = max(1, grid_height - y)
            
            layout[dept_id] = (x, y, width, height)
            x += width
        
        return layout
    
    def swap_departments(self, layout, dept1, dept2):
        """Troca as posições de dois departamentos"""
        new_layout = layout.copy()
        new_layout[dept1], new_layout[dept2] = layout[dept2], layout[dept1]
        return new_layout
    
    def optimize(self, initial_layout, max_iterations=50):
        """
        Executa o algoritmo CRAFT
        Retorna o layout otimizado e histórico de custos
        """
        current_layout = initial_layout.copy()
        current_cost = self.calculate_total_cost(current_layout)
        
        self.history = [current_cost]
        self.swap_history = []
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            dept_ids = list(self.departments.keys())
            
            # Testa todas as combinações de pares de departamentos
            for dept1, dept2 in combinations(dept_ids, 2):
                new_layout = self.swap_departments(current_layout, dept1, dept2)
                new_cost = self.calculate_total_cost(new_layout)
                
                # Se houve melhoria, aceita a troca
                if new_cost < current_cost:
                    cost_reduction = current_cost - new_cost
                    self.swap_history.append({
                        'iteration': iteration,
                        'dept1': self.departments[dept1]['name'],
                        'dept2': self.departments[dept2]['name'],
                        'cost_before': current_cost,
                        'cost_after': new_cost,
                        'reduction': cost_reduction
                    })
                    current_layout = new_layout
                    current_cost = new_cost
                    improved = True
                    self.history.append(current_cost)
                    break
            
            iteration += 1
        
        return current_layout, current_cost, self.history


def plot_layout(layout, departments, flows, title, colors):
    """Função para plotar um layout com legenda e visualização de fluxos"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    patches_list = []
    
    # 1. Desenhar Departamentos
    for dept_id, (x, y, w, h) in layout.items():
        dept_name = departments[dept_id]['name']
        
        # Desenhar retângulo
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                              linewidth=2.5, edgecolor='black', 
                              facecolor=colors[dept_id], alpha=0.8,
                              label=dept_name) # Adiciona label para a legenda
        ax.add_patch(rect)
        patches_list.append(rect)
        
        # Adicionar texto
        ax.text(x + w/2, y + h/2, dept_name, 
               ha='center', va='center', fontsize=11, fontweight='bold', color='black',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
    
    # 2. Desenhar Fluxos (apenas os 3 maiores para não poluir)
    if flows and title == "Layout Otimizado": # Apenas no layout otimizado
        flow_list = []
        for (i, j), flow in flows.items():
            if i < j: # Evita duplicidade
                flow_list.append(((i, j), flow))
        
        # Ordena os fluxos do maior para o menor
        flow_list.sort(key=lambda x: x[1], reverse=True)
        top_flows = flow_list[:3]
        
        for (i, j), flow in top_flows:
            x1, y1, w1, h1 = layout[i]
            x2, y2, w2, h2 = layout[j]
            
            center1 = (x1 + w1/2, y1 + h1/2)
            center2 = (x2 + w2/2, y2 + h2/2)
            
            # Desenhar seta para representar o fluxo
            arrow = FancyArrowPatch(center1, center2, 
                                    arrowstyle='simple,head_width=8,head_length=8', 
                                    color='red', 
                                    linewidth=1 + (flow / 100), # Espessura proporcional ao fluxo
                                    alpha=0.6,
                                    mutation_scale=10)
            ax.add_patch(arrow)

    # 3. Configurar Eixos e Título
    if layout:
        max_x = max(x + w for x, y, w, h in layout.values())
        max_y = max(y + h for x, y, w, h in layout.values())
        ax.set_xlim(-2, max_x + 2)
        ax.set_ylim(-2, max_y + 2)
    
    ax.set_aspect('equal')
    ax.set_xlabel('Coordenada X', fontsize=10)
    ax.set_ylabel('Coordenada Y', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Adicionar Legenda
    labels = [p.get_label() for p in patches_list]
    ax.legend(patches_list, labels, loc='upper right', bbox_to_anchor=(1.25, 1.0))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta o layout para acomodar a legenda
    return fig


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

# Inicializar session state
if 'departments' not in st.session_state:
    st.session_state.departments = {}
if 'flows' not in st.session_state:
    st.session_state.flows = {}

# Abas principais
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dados", "⚙️ Otimização", "📈 Resultados", "📋 Análise", "ℹ️ Sobre"])

with tab1:
    st.header("📊 Dados de Entrada")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("🏢 Departamentos")
        num_depts = st.slider("Número de departamentos:", 3, 12, 5, key="num_depts")
        
        departments = {}
        dept_data = []
        
        cols = st.columns(3)
        for i in range(num_depts):
            with cols[i % 3]:
                st.write(f"**Depto {i+1}**")
                dept_name = st.text_input(f"Nome:", value=f"Depto_{i+1}", key=f"dept_name_{i}", label_visibility="collapsed")
                dept_area = st.number_input(f"Área (m²):", value=100, min_value=10, key=f"dept_area_{i}", label_visibility="collapsed")
                
                departments[i] = {'name': dept_name, 'area': dept_area}
                dept_data.append({'ID': i, 'Nome': dept_name, 'Área (m²)': dept_area})
        
        st.dataframe(pd.DataFrame(dept_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🔄 Fluxo de Materiais")
        st.markdown("Defina o fluxo (unidades/período) entre departamentos (Matriz de Fluxo):")
        
        dept_names = [departments[i]['name'] for i in range(num_depts)]
        
        # Inicializar a matriz de fluxo
        if 'flow_df' not in st.session_state or st.session_state.flow_df.shape[0] != num_depts:
            initial_flow_data = np.full((num_depts, num_depts), 10)
            np.fill_diagonal(initial_flow_data, 0)
            st.session_state.flow_df = pd.DataFrame(initial_flow_data, index=dept_names, columns=dept_names)
        else:
            # Atualizar nomes de colunas/índices se o número de departamentos for o mesmo
            st.session_state.flow_df.index = dept_names
            st.session_state.flow_df.columns = dept_names
            
        # Usar st.data_editor para edição interativa
        edited_flow_df = st.data_editor(
            st.session_state.flow_df,
            column_config={
                col: st.column_config.NumberColumn(
                    f"Fluxo para {col}",
                    min_value=0,
                    default=10,
                    format="%d"
                )
                for col in dept_names
            },
            hide_index=False,
            use_container_width=True,
            key="flow_editor"
        )
        
        # Processar o DataFrame editado para o formato de dicionário de fluxos
        flows = {}
        for i in range(num_depts):
            for j in range(i + 1, num_depts):
                dept1_name = dept_names[i]
                dept2_name = dept_names[j]
                
                # O fluxo é simétrico, pegamos o valor da célula (i, j)
                flow_value = edited_flow_df.loc[dept1_name, dept2_name]
                
                flows[(i, j)] = flow_value
                flows[(j, i)] = flow_value
        
        # Armazenar no session state
        st.session_state.departments = departments
        st.session_state.flows = flows
        st.session_state.flow_df = edited_flow_df # Atualiza o estado para persistência

with tab2:
    st.header("⚙️ Configuração da Otimização")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        grid_width = st.slider("Largura da grade:", 10, 50, 20, key="grid_width")
    
    with col2:
        grid_height = st.slider("Altura da grade:", 10, 50, 20, key="grid_height")
    
    with col3:
        arrangement = st.selectbox(
            "Arranjo inicial:",
            ["random", "ordered", "by_area"],
            format_func=lambda x: {
                "random": "🎲 Aleatório",
                "ordered": "📋 Ordenado",
                "by_area": "📐 Por Área"
            },
            key="arrangement"
        )
    
    with col4:
        distance_metric = st.selectbox(
            "Métrica de distância:",
            ["manhattan", "euclidean"],
            format_func=lambda x: {"manhattan": "Manhattan", "euclidean": "Euclidiana"}[x],
            key="distance_metric"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_iterations = st.slider("Máximo de iterações:", 10, 200, 50, key="max_iterations")
    
    with col2:
        st.info("💡 Mais iterações = melhor otimização, mas mais tempo de processamento")
    
    # Botão de execução
    if st.button("🚀 Executar Otimização CRAFT", use_container_width=True, type="primary"):
        if not st.session_state.departments or not st.session_state.flows:
            st.error("❌ Por favor, configure os departamentos e fluxos na aba 'Dados'")
        else:
            with st.spinner("⏳ Otimizando layout... Isso pode levar alguns segundos..."):
                # Criar otimizador
                optimizer = CRAFTOptimizer(
                    st.session_state.departments,
                    st.session_state.flows,
                    st.session_state.distance_metric
                )
                
                # Gerar layout inicial
                initial_layout = optimizer.generate_initial_layout(
                    st.session_state.grid_width,
                    st.session_state.grid_height,
                    st.session_state.arrangement
                )
                initial_cost = optimizer.calculate_total_cost(initial_layout)
                
                # Otimizar
                optimized_layout, optimized_cost, history = optimizer.optimize(
                    initial_layout,
                    st.session_state.max_iterations
                )
                
                # Armazenar no session state
                st.session_state.optimizer = optimizer
                st.session_state.initial_layout = initial_layout
                st.session_state.optimized_layout = optimized_layout
                st.session_state.initial_cost = initial_cost
                st.session_state.optimized_cost = optimized_cost
                st.session_state.history = history
                st.session_state.optimization_time = datetime.now()
                
                st.success("✅ Otimização concluída com sucesso!")

with tab3:
    st.header("📈 Resultados da Otimização")
    
    if 'optimized_layout' not in st.session_state:
        st.info("👈 Execute a otimização na aba '⚙️ Otimização' para ver os resultados.")
    else:
        optimizer = st.session_state.optimizer
        initial_layout = st.session_state.initial_layout
        optimized_layout = st.session_state.optimized_layout
        initial_cost = st.session_state.initial_cost
        optimized_cost = st.session_state.optimized_cost
        history = st.session_state.history
        depts = st.session_state.departments
        
        # Métricas principais
        st.subheader("📊 Métricas Principais")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Custo Inicial", f"{initial_cost:.2f}")
        
        with col2:
            st.metric("✨ Custo Otimizado", f"{optimized_cost:.2f}")
        
        with col3:
            reduction = ((initial_cost - optimized_cost) / initial_cost * 100) if initial_cost > 0 else 0
            st.metric("📉 Redução de Custo", f"{reduction:.2f}%", delta=f"-{reduction:.2f}%")
        
        with col4:
            st.metric("🔄 Iterações", len(history) - 1)
        
        st.divider()
        
        # Cores para os departamentos
        colors = plt.cm.Set3(np.linspace(0, 1, len(depts)))
        
        # Visualização dos layouts
        st.subheader("🎨 Comparação de Layouts")
        
        col1, col2 = st.columns(2)
        
        # Cores para os departamentos
        colors = plt.cm.Set3(np.linspace(0, 1, len(depts)))
        
        with col1:
            fig = plot_layout(initial_layout, depts, {}, "Layout Inicial", colors) # Passa flows vazio para o inicial
            st.pyplot(fig)
        
        with col2:
            fig = plot_layout(optimized_layout, depts, st.session_state.flows, "Layout Otimizado", colors) # Passa flows para o otimizado
            st.pyplot(fig)
        
        st.divider()
        
        # Gráfico de convergência
        st.subheader("📊 Convergência do Algoritmo")
        
        fig, ax = plt.subplots(figsize=(14, 5))
        iterations = range(len(history))
        ax.plot(iterations, history, marker='o', linewidth=2.5, markersize=5, color='#1f77b4', label='Custo Total')
        ax.fill_between(iterations, history, alpha=0.3, color='#1f77b4')
        
        ax.set_xlabel('Iteração', fontsize=11, fontweight='bold')
        ax.set_ylabel('Custo Total de Movimentação', fontsize=11, fontweight='bold')
        ax.set_title('Redução de Custo ao Longo das Iterações', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        
        # Tabela de comparação
        st.subheader("📋 Comparação de Posições dos Departamentos")
        
        comparison_data = []
        for dept_id in depts.keys():
            x1, y1, w1, h1 = initial_layout[dept_id]
            x2, y2, w2, h2 = optimized_layout[dept_id]
            moved = (x1, y1) != (x2, y2)
            comparison_data.append({
                'Departamento': depts[dept_id]['name'],
                'Pos. Inicial': f"({x1}, {y1})",
                'Pos. Otimizada': f"({x2}, {y2})",
                'Mudou': '✓ Sim' if moved else '✗ Não',
                'Distância': f"{np.sqrt((x2-x1)**2 + (y2-y1)**2):.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

with tab4:
    st.header("📋 Análise Detalhada")
    
    if 'optimized_layout' not in st.session_state:
        st.info("👈 Execute a otimização na aba '⚙️ Otimização' para ver a análise.")
    else:
        optimizer = st.session_state.optimizer
        depts = st.session_state.departments
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Histórico de Trocas")
            
            if optimizer.swap_history:
                swap_df = pd.DataFrame(optimizer.swap_history)
                swap_df['Redução'] = swap_df['reduction'].apply(lambda x: f"{x:.2f}")
                swap_df['Custo Antes'] = swap_df['cost_before'].apply(lambda x: f"{x:.2f}")
                swap_df['Custo Depois'] = swap_df['cost_after'].apply(lambda x: f"{x:.2f}")
                
                display_df = swap_df[['iteration', 'dept1', 'dept2', 'Custo Antes', 'Custo Depois', 'Redução']]
                display_df.columns = ['Iter', 'Depto 1', 'Depto 2', 'Custo Antes', 'Custo Depois', 'Redução']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhuma troca foi realizada (layout já estava otimizado)")
        
        with col2:
            st.subheader("📈 Estatísticas da Otimização")
            
            stats_data = {
                'Métrica': [
                    'Total de Iterações',
                    'Total de Trocas',
                    'Redução Total de Custo',
                    'Redução Percentual',
                    'Métrica de Distância',
                    'Arranjo Inicial'
                ],
                'Valor': [
                    len(st.session_state.history) - 1,
                    len(optimizer.swap_history),
                    f"{st.session_state.initial_cost - st.session_state.optimized_cost:.2f}",
                    f"{((st.session_state.initial_cost - st.session_state.optimized_cost) / st.session_state.initial_cost * 100):.2f}%",
                    st.session_state.distance_metric.capitalize(),
                    st.session_state.arrangement.capitalize()
                ]
            }
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("🔍 Análise de Fluxos")
        
        # Calcular fluxos mais significativos
        flows_sorted = sorted(
            st.session_state.flows.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Remover duplicatas (i,j) e (j,i)
        unique_flows = []
        seen = set()
        for (i, j), flow in flows_sorted:
            if (min(i, j), max(i, j)) not in seen:
                unique_flows.append(((i, j), flow))
                seen.add((min(i, j), max(i, j)))
        
        top_flows = unique_flows[:5]
        
        flow_analysis = []
        for (i, j), flow in top_flows:
            if flow > 0:
                initial_dist = optimizer.calculate_distance(
                    (initial_layout[i][0] + initial_layout[i][2]/2, initial_layout[i][1] + initial_layout[i][3]/2),
                    (initial_layout[j][0] + initial_layout[j][2]/2, initial_layout[j][1] + initial_layout[j][3]/2)
                )
                optimized_dist = optimizer.calculate_distance(
                    (optimized_layout[i][0] + optimized_layout[i][2]/2, optimized_layout[i][1] + optimized_layout[i][3]/2),
                    (optimized_layout[j][0] + optimized_layout[j][2]/2, optimized_layout[j][1] + optimized_layout[j][3]/2)
                )
                
                flow_analysis.append({
                    'Fluxo': f"{depts[i]['name']} → {depts[j]['name']}",
                    'Quantidade': flow,
                    'Dist. Inicial': f"{initial_dist:.2f}",
                    'Dist. Otimizada': f"{optimized_dist:.2f}",
                    'Melhoria': f"{((initial_dist - optimized_dist) / initial_dist * 100):.1f}%" if initial_dist > 0 else "0%"
                })
        
        st.dataframe(pd.DataFrame(flow_analysis), use_container_width=True, hide_index=True)

with tab5:
    st.header("ℹ️ Sobre o CRAFT")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🏭 O que é CRAFT?
        
        **CRAFT** (Computerized Relative Allocation of Facilities Technique) é um algoritmo clássico de otimização 
        de layouts industriais que busca **minimizar o custo total de movimentação** de materiais entre departamentos.
        
        Desenvolvido por Armour e Buffa em 1963, o CRAFT é um dos algoritmos mais utilizados em pesquisa operacional 
        e engenharia industrial para resolver problemas de arranjo físico.
        
        ### 📐 Função Objetivo
        
        O algoritmo minimiza o custo total calculado como:
        
        ```
        Custo Total = Σ (Fᵢⱼ × Dᵢⱼ × C)
        ```
        
        Onde:
        - **Fᵢⱼ**: Fluxo de materiais entre departamentos i e j
        - **Dᵢⱼ**: Distância entre os centros dos departamentos i e j
        - **C**: Custo unitário por unidade de distância
        
        ### 🔄 Como Funciona
        
        1. **Gera um layout inicial** (aleatório, ordenado ou por área)
        2. **Calcula o custo total** do layout atual
        3. **Testa trocas de pares de departamentos** para encontrar melhorias
        4. **Aceita a troca** se reduzir o custo total
        5. **Repete** até atingir um ótimo local ou máximo de iterações
        
        ### 📏 Métricas de Distância
        
        **Manhattan (Distância de Bloco):**
        ```
        D = |x₁ - x₂| + |y₁ - y₂|
        ```
        Simula movimento em grade (comum em fábricas com corredores)
        
        **Euclidiana (Linha Reta):**
        ```
        D = √((x₁ - x₂)² + (y₁ - y₂)²)
        ```
        Simula movimento direto (comum em armazéns abertos)
        
        ### 💼 Aplicações Práticas
        
        Este MicroSaaS pode ser usado para otimizar layouts de:
        - 🏭 Fábricas e plantas industriais
        - 📦 Armazéns e centros de distribuição
        - 🏥 Hospitais e clínicas
        - 🏢 Escritórios e centros de trabalho
        - 🛒 Lojas e centros comerciais
        - 🔧 Oficinas e centros de serviço
        
        ### ⚡ Vantagens do CRAFT
        
        - ✅ Algoritmo simples e intuitivo
        - ✅ Converge rapidamente para soluções boas
        - ✅ Fácil de implementar e compreender
        - ✅ Não requer conhecimento matemático avançado
        - ✅ Resultados práticos e aplicáveis
        
        ### ⚠️ Limitações
        
        - ⚠️ Encontra ótimos locais, não globais
        - ⚠️ Sensível ao layout inicial
        - ⚠️ Não considera restrições físicas
        - ⚠️ Assume departamentos retangulares
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Referências
        
        **Autores Originais:**
        - Armour, G. C.
        - Buffa, E. S.
        
        **Ano:** 1963
        
        **Publicação:**
        Journal of Industrial Engineering
        
        ### 🎯 Próximos Passos
        
        Melhorias futuras podem incluir:
        
        - 🔄 Algoritmos genéticos
        - 🌡️ Simulated annealing
        - 🐜 Otimização por colônia de formigas
        - 🚫 Restrições físicas
        - 🎨 Formatos não-retangulares
        - 📊 Análise de sensibilidade
        
        ### 📞 Suporte
        
        Para dúvidas ou sugestões sobre este MicroSaaS, 
        entre em contato com a equipe de desenvolvimento.
        
        ---
        
        **CRAFT Layout Optimizer v1.0**
        
        MicroSaaS Project | 2025
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 11px; padding: 20px;'>
    <p>🏭 CRAFT Layout Optimizer v1.0 | MicroSaaS Project | 2025</p>
    <p>Desenvolvido com ❤️ usando Streamlit | Algoritmo CRAFT © Armour & Buffa (1963)</p>
</div>
""", unsafe_allow_html=True)
