"""
Help Acsa Silveira - Métodos Numéricos Computacionais

Este módulo contém implementações de diversos métodos numéricos para:
- Encontrar raízes de funções
- Ajuste de curvas e regressão
- Interpolação polinomial
- Integração numérica
- Solução de sistemas lineares
- Decomposição matricial

Autor: Acsa Silveira
Data: 16/07/2025
"""
# ==============================================
# IMPORTAÇÕES
# ==============================================

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression


# ==============================================
# MÉTODOS PARA ENCONTRAR RAÍZES DE FUNÇÕES
# ==============================================

""" Descrição da Função: 

Plota o gráfico de uma função matemática no intervalo especificado.
    
    Parâmetros:
    -----------
    funcao : function
        Função a ser plotada (deve aceitar operações vetoriais)
    inicio_intervalo : float
        Início do intervalo do domínio
    fim_intervalo : float
        Fim do intervalo do domínio
    qtd_pontos : int, opcional
        Número de pontos para plotagem (padrão=1000)
        
    Retorna:
    --------
    None (exibe o gráfico) """
def grafico(funcao, inicio_intervalo, fim_intervalo, qtd_pontos=1000):
    x = np.linspace(inicio_intervalo, fim_intervalo, qtd_pontos)
    y = funcao(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("Gráfico de f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

""" Descrição da Função:

Método da bisseção para encontrar raízes de funções.
    
    Parâmetros:
    -----------
    f : function
        Função contínua onde será buscada a raiz
    a : float
        Extremo inferior do intervalo inicial
    b : float
        Extremo superior do intervalo inicial
    tol : float
        Tolerância para critério de parada
    plot_iteracoes : bool, opcional
        Se True, plota gráfico de convergência (padrão=False)
    retornar_dados : bool, opcional
        Se True, retorna listas de iterações e erros (padrão=False)
        
    Retorna:
    --------
    tuple or None:
        - Se retornar_dados=False: imprime relatório e mostra gráfico (se plot_iteracoes=True)
        - Se retornar_dados=True: retorna (iteracoes, erros)
"""
def bissecao(f, a, b, tol, plot_iteracoes=False, retornar_dados=False):
    if f(a) * f(b) >= 0:
        raise ValueError("A função deve ter sinais opostos nos extremos do intervalo")
    
    iteracoes = []
    erros = []
    i = 0
    
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        iteracoes.append(i)
        erros.append(abs(f(c)))
        i += 1
        
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    raiz = (a + b) / 2
    
    if retornar_dados:
        return iteracoes, erros
    else:
        print(f"Raiz encontrada: {raiz:.8f}")
        print(f"Número de iterações: {i}")
        print(f"Erro final: {abs(f(raiz)):.2e}")
        
        if plot_iteracoes:
            plt.figure(figsize=(10, 5))
            plt.plot(iteracoes, erros, 'o-')
            plt.xlabel('Iteração')
            plt.ylabel('|f(x)|')
            plt.title('Convergência do Método da Bisseção')
            plt.grid(True)
            plt.show()

""" Descrição da Função:

Método da secante para encontrar raízes de funções.
    
    Parâmetros:
    -----------
    f : function
        Função onde será buscada a raiz
    x0, x1 : float
        Valores iniciais para o método
    tol : float, opcional
        Tolerância para critério de parada (padrão=1e-6)
    max_iter : int, opcional
        Número máximo de iterações (padrão=100)
    plot_iteracoes : bool, opcional
        Se True, plota gráfico de convergência (padrão=False)
    retornar_dados : bool, opcional
        Se True, retorna listas de iterações e erros (padrão=False)
        
    Retorna:
    --------
    tuple or float:
        - Se retornar_dados=True: retorna (iteracoes, erros)
        - Caso contrário: retorna a aproximação da raiz ou None se não convergir
"""
def secante(f, x0, x1, tol=1e-6, max_iter=100, plot_iteracoes=False, retornar_dados=False):
    iteracoes = []
    erros = []
    
    for i in range(max_iter):
        fx1 = f(x1)
        fx0 = f(x0)
        
        if abs(fx1) < tol:
            if retornar_dados:
                return iteracoes, erros
            else:
                print(f"Raiz encontrada: {x1:.8f}")
                print(f"Número de iterações: {i+1}")
                print(f"Erro final: {abs(fx1):.2e}")
                
                if plot_iteracoes:
                    plt.figure(figsize=(10, 5))
                    plt.plot(iteracoes, erros, 'o-')
                    plt.xlabel('Iteração')
                    plt.ylabel('|f(x)|')
                    plt.title('Convergência do Método da Secante')
                    plt.grid(True)
                    plt.show()
                return x1
        
        if abs(fx1 - fx0) < 1e-14:
            print("Divisão por zero evitada - método parado")
            return None
            
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        iteracoes.append(i)
        erros.append(abs(fx1))
        
        x0, x1 = x1, x_new
    
    print("Método não convergiu após o número máximo de iterações")
    return None

""" Descrição da Função:

Método da falsa posição para encontrar raízes de funções.
    
    Parâmetros:
    -----------
    f : function
        Função contínua onde será buscada a raiz
    a : float
        Extremo inferior do intervalo inicial
    b : float
        Extremo superior do intervalo inicial
    tol : float, opcional
        Tolerância para critério de parada (padrão=1e-6)
    max_iter : int, opcional
        Número máximo de iterações (padrão=100)
    plot_iteracoes : bool, opcional
        Se True, plota gráfico de convergência (padrão=False)
    retornar_dados : bool, opcional
        Se True, retorna listas de iterações e erros (padrão=False)
        
    Retorna:
    --------
    tuple or float:
        - Se retornar_dados=True: retorna (iteracoes, erros)
        - Caso contrário: retorna a aproximação da raiz ou None se não convergir
"""
def falsa_posicao(f, a, b, tol=1e-6, max_iter=100, plot_iteracoes=False, retornar_dados=False):
    if f(a) * f(b) >= 0:
        raise ValueError("A função deve ter sinais opostos nos extremos do intervalo")
    
    iteracoes = []
    erros = []
    
    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        fc = f(c)
        iteracoes.append(i)
        erros.append(abs(fc))
        
        if abs(fc) < tol:
            if retornar_dados:
                return iteracoes, erros
            else:
                print(f"Raiz encontrada: {c:.8f}")
                print(f"Número de iterações: {i+1}")
                print(f"Erro final: {abs(fc):.2e}")
                
                if plot_iteracoes:
                    plt.figure(figsize=(10, 5))
                    plt.plot(iteracoes, erros, 'o-')
                    plt.xlabel('Iteração')
                    plt.ylabel('|f(x)|')
                    plt.title('Convergência do Método da Falsa Posição')
                    plt.grid(True)
                    plt.show()
                return c
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    print("Método não convergiu após o número máximo de iterações")
    return None

""" Descrição da Função:

Compara visualmente a convergência dos métodos de busca de raízes.
    
    Parâmetros:
    -----------
    f : function
        Função contínua onde será buscada a raiz
    a : float
        Extremo inferior do intervalo inicial
    b : float
        Extremo superior do intervalo inicial
    tol : float, opcional
        Tolerância para critério de parada (padrão=1e-6)
        
    Retorna:
    --------
    None (exibe gráfico comparativo)

"""
def comparar_metodos(f, a, b, tol=1e-6):
    # Obter dados de convergência de cada método
    iter_bis, err_bis = bissecao(f, a, b, tol, retornar_dados=True)
    iter_sec, err_sec = secante(f, a, b, tol, retornar_dados=True)
    iter_fp, err_fp = falsa_posicao(f, a, b, tol, retornar_dados=True)
    
    # Plotar comparação
    plt.figure(figsize=(10, 6))
    plt.semilogy(iter_bis, err_bis, 'o-', label='Bisseção')
    plt.semilogy(iter_sec, err_sec, 's-', label='Secante')
    plt.semilogy(iter_fp, err_fp, 'd-', label='Falsa Posição')
    
    plt.xlabel('Iteração')
    plt.ylabel('Erro |f(x)| (escala log)')
    plt.title('Comparação da Convergência dos Métodos')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()


# ==============================================
# MÉTODOS PARA AJUSTE DE CURVAS E REGRESSÃO
# ==============================================

""" Descrição da Função: 

Ajusta uma reta pelo método dos mínimos quadrados.
    
    Parâmetros:
    -----------
    x : array_like
        Valores das abscissas
    y : array_like
        Valores das ordenadas
    retornar_coef : bool, opcional
        Se True, retorna coeficientes (a, b) em vez da equação (padrão=False)
    plot : bool, opcional
        Se True, plota os dados e a reta ajustada (padrão=False)
        
    Retorna:
    --------
    str or tuple:
        - Se retornar_coef=False: string com equação da reta
        - Se retornar_coef=True: tupla com coeficientes (a, b)
    """
def minimos_quadrados(x, y, retornar_coef=False, plot=False):
    n = len(x)
    x = np.asarray(x)
    y = np.asarray(y)
    
    soma_x = x.sum()
    soma_y = y.sum()
    soma_xy = (x * y).sum()
    soma_x2 = (x**2).sum()
    
    a = (n * soma_xy - soma_x * soma_y) / (n * soma_x2 - soma_x**2)
    b = (soma_y - a * soma_x) / n
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, label='Dados')
        plt.plot(x, a*x + b, 'r', label=f'Ajuste: y = {a:.4f}x + {b:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title('Ajuste Linear por Mínimos Quadrados')
        plt.show()
    
    if retornar_coef:
        return a, b
    else:
        return f"y = {a:.4f}x + {b:.4f}"

""" Descrição da Função:

Calcula o coeficiente de determinação R² para um ajuste polinomial.
    
    Parâmetros:
    -----------
    x : array_like
        Valores das abscissas
    y : array_like
        Valores das ordenadas
    grau : int, opcional
        Grau do polinômio de ajuste (padrão=2)
    plot : bool, opcional
        Se True, plota os dados e a curva ajustada (padrão=False)
        
    Retorna:
    --------
    tuple:
        (R², equacao_str)
    """
def coeficiente_determinacao(x, y, grau=2, plot=False):
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Ajuste polinomial
    coef = np.polyfit(x, y, grau)
    p = np.poly1d(coef)
    
    # Cálculo de R²
    y_pred = p(x)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Formata a equação
    termos = []
    for i, c in enumerate(coef):
        exp = len(coef) - i - 1
        if exp == 0:
            termos.append(f"{c:.4f}")
        elif exp == 1:
            termos.append(f"{c:.4f}x")
        else:
            termos.append(f"{c:.4f}x^{exp}")
    eq = "y = " + " + ".join(termos)
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, label='Dados')
        x_vals = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_vals, p(x_vals), 'r', label=f'Ajuste (R² = {r2:.4f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title(f'Ajuste Polinomial (grau {grau})')
        plt.show()
    
    return r2, eq

""" Descrição da Função:

Realiza regressão linear simples e calcula R².
    
    Parâmetros:
    -----------
    x : array_like
        Valores das abscissas
    y : array_like
        Valores das ordenadas
    plot : bool, opcional
        Se True, plota os dados e a reta ajustada (padrão=False)
        
    Retorna:
    --------
    tuple:
        (funcao_ajuste, R²)
    """
def regressao_linear(x, y, plot=False):
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    cov = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean)**2)
    
    a = cov / var_x
    b = y_mean - a * x_mean
    
    # Função de ajuste
    def ajuste(x_val):
        return a * x_val + b
    
    # Cálculo de R²
    y_pred = ajuste(x)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, label='Dados')
        plt.plot(x, ajuste(x), 'r', label=f'Ajuste (R² = {r2:.4f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title('Regressão Linear Simples')
        plt.show()
    
    return ajuste, r2

""" Descrição da Função:

Realiza regressão polinomial e calcula R².
    
    Parâmetros:
    -----------
    x : array_like
        Valores das abscissas
    y : array_like
        Valores das ordenadas
    grau : int
        Grau do polinômio de ajuste
    plot : bool, opcional
        Se True, plota os dados e a curva ajustada (padrão=False)
        
    Retorna:
    --------
    tuple:
        (funcao_ajuste, R²)
    """
def regressao_polinomial(x, y, grau, plot=False):
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Ajuste polinomial
    coef = np.polyfit(x, y, grau)
    p = np.poly1d(coef)
    
    # Cálculo de R²
    y_pred = p(x)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, label='Dados')
        x_vals = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_vals, p(x_vals), 'r', label=f'Ajuste (R² = {r2:.4f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title(f'Regressão Polinomial (grau {grau})')
        plt.show()
    
    return p, r2

""" Descrição da Função:

Interpolação polinomial usando o método de Lagrange.
    
    Parâmetros:
    -----------
    x : array_like
        Valores das abscissas conhecidos
    y : array_like
        Valores das ordenadas conhecidos
    ponto : float, opcional
        Ponto para avaliar o polinômio interpolador (padrão=None)
    plot : bool, opcional
        Se True, plota os pontos e o polinômio interpolador (padrão=False)
        
    Retorna:
    --------
    float or tuple:
        - Se ponto for fornecido: valor interpolado no ponto
        - Caso contrário: função do polinômio interpolador
    """
def interpola_lagrange(x, y, ponto=None, plot=False):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    
    def polinomio_lagrange(x_val):
        resultado = 0.0
        for i in range(n):
            termo = y[i]
            for j in range(n):
                if j != i:
                    termo *= (x_val - x[j]) / (x[i] - x[j])
            resultado += termo
        return resultado
    
    if ponto is not None:
        return polinomio_lagrange(ponto)
    
    if plot:
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = [polinomio_lagrange(xi) for xi in x_vals]
        
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, label='Pontos conhecidos')
        plt.plot(x_vals, y_vals, 'r', label='Polinômio de Lagrange')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title('Interpolação de Lagrange')
        plt.show()
    
    return polinomio_lagrange

""" Descrição da Função:

Calcula a tabela de diferenças divididas para interpolação de Newton.
    
    Parâmetros:
    -----------
    x : array_like
        Valores das abscissas
    y : array_like
        Valores das ordenadas
        
    Retorna:
    --------
    array:
        Tabela de diferenças divididas (triangular)
    """
def diferencas_divididas(x, y):
    
    n = len(x)
    tabela = np.zeros((n, n))
    tabela[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            tabela[i, j] = (tabela[i+1, j-1] - tabela[i, j-1]) / (x[i+j] - x[i])
    
    return tabela

""" Descrição da Função:

Interpolação polinomial usando o método de Newton.
    
    Parâmetros:
    -----------
    x : array_like
        Valores das abscissas conhecidos
    y : array_like
        Valores das ordenadas conhecidos
    ponto : float, opcional
        Ponto para avaliar o polinômio interpolador (padrão=None)
    plot : bool, opcional
        Se True, plota os pontos e o polinômio interpolador (padrão=False)
        
    Retorna:
    --------
    float or tuple:
        - Se ponto for fornecido: valor interpolado no ponto
        - Caso contrário: função do polinômio interpolador
    """
def interpola_newton(x, y, ponto=None, plot=False):
    
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    
    # Calcula diferenças divididas
    tabela = diferencas_divididas(x, y)
    coef = tabela[0, :]  # Coeficientes do polinômio
    
    # Constrói o polinômio de Newton
    def polinomio_newton(x_val):
        resultado = coef[0]
        produto = 1.0
        for i in range(1, n):
            produto *= (x_val - x[i-1])
            resultado += coef[i] * produto
        return resultado
    
    if ponto is not None:
        return polinomio_newton(ponto)
    
    if plot:
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = [polinomio_newton(xi) for xi in x_vals]
        
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, label='Pontos conhecidos')
        plt.plot(x_vals, y_vals, 'r', label='Polinômio de Newton')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title('Interpolação de Newton')
        plt.show()
    
    return polinomio_newton


# ==============================================
# MÉTODOS PARA INTEGRAÇÃO NUMÉRICA
# ==============================================

""" Descrição da Função:

Calcula a integral definida usando a regra dos trapézios.
    
    Parâmetros:
    -----------
    f : function
        Função a ser integrada
    a : float
        Limite inferior de integração
    b : float
        Limite superior de integração
    n : int, opcional
        Número de subintervalos (padrão=1000)
        
    Retorna:
    --------
    float:
        Valor aproximado da integral
    """
def integral_trapezios(f, a, b, n=1000):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2
    
    for i in range(1, n):
        integral += f(a + i * h)
    
    return integral * h

""" Descrição da Função:

Calcula a integral definida usando a regra de Simpson.
    
    Parâmetros:
    -----------
    f : function
        Função a ser integrada
    a : float
        Limite inferior de integração
    b : float
        Limite superior de integração
    n : int, opcional
        Número de subintervalos (deve ser par, padrão=1000)
        
    Retorna:
    --------
    float:
        Valor aproximado da integral
    """
def integral_simpson(f, a, b, n=1000):
    if n % 2 != 0:
        n += 1  # Garante que n seja par
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])  # Soma dos termos ímpares
    integral += 2 * np.sum(y[2:-2:2])  # Soma dos termos pares
    
    return integral * h / 3

""" Descrição da Função:

Compara os métodos de integração numérica.
    
    Parâmetros:
    -----------
    f : function
        Função a ser integrada
    a : float
        Limite inferior de integração
    b : float
        Limite superior de integração
    n : int, opcional
        Número de subintervalos (padrão=1000)
    plot : bool, opcional
        Se True, plota a função e a aproximação (padrão=False)
        
    Retorna:
    --------
    dict:
        Resultados dos diferentes métodos
    """
def comparar_integrais(f, a, b, n=1000, plot=False):
    resultado_trap = integral_trapezios(f, a, b, n)
    resultado_simp = integral_simpson(f, a, b, n)
    
    resultados = {
        'Trapézios': resultado_trap,
        'Simpson': resultado_simp,
        'Diferença': abs(resultado_trap - resultado_simp)
    }
    
    if plot:
        x = np.linspace(a, b, n+1)
        y = f(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='Função')
        
        # Área sob os trapézios
        plt.fill_between(x, y, color='lightblue', alpha=0.3, label='Área (Trapézios)')
        
        # Pontos de Simpson
        if n <= 20:  # Só mostra pontos se n for pequeno
            plt.plot(x[::2], y[::2], 'ro', label='Pontos de Simpson')
        
        plt.title(f'Integração Numérica\nTrapézios: {resultado_trap:.6f} | Simpson: {resultado_simp:.6f}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return resultados


# ==============================================
# MÉTODOS PARA SISTEMAS LINEARES
# ==============================================

""" Descrição da Função:

Resolve um sistema triangular inferior Lx = b.
    
    Parâmetros:
    -----------
    L : array_like
        Matriz triangular inferior (n x n)
    b : array_like
        Vetor de termos independentes (n)
        
    Retorna:
    --------
    array:
        Vetor solução x
    """
def substituicao_sucessiva(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    
    for i in range(n):
        soma = np.dot(L[i, :i], x[:i])
        x[i] = (b[i] - soma) / L[i, i]
    
    return x

""" Descrição da Função:

Resolve um sistema triangular superior Ux = b.
    
    Parâmetros:
    -----------
    U : array_like
        Matriz triangular superior (n x n)
    b : array_like
        Vetor de termos independentes (n)
        
    Retorna:
    --------
    array:
        Vetor solução x
    """
def substituicao_retroativa(U, b):
    
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        soma = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (b[i] - soma) / U[i, i]
    
    return x

""" Descrição da Função:

Resolve um sistema linear Ax = b por eliminação de Gauss.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz de coeficientes (n x n)
    b : array_like
        Vetor de termos independentes (n)
        
    Retorna:
    --------
    array:
        Vetor solução x
    """
def eliminacao_gauss(A, b):
    n = A.shape[0]
    A = A.astype(float)
    b = b.astype(float)
    
    # Fase de eliminação
    for k in range(n-1):
        # Pivotamento parcial
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        
        # Eliminação
        for i in range(k+1, n):
            fator = A[i, k] / A[k, k]
            A[i, k:] -= fator * A[k, k:]
            b[i] -= fator * b[k]
    
    # Fase de substituição retroativa
    return substituicao_retroativa(A, b)

""" Descrição da Função:

Realiza a decomposição LU de uma matriz.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz a ser decomposta (n x n)
        
    Retorna:
    --------
    tuple:
        (L, U) - Matrizes triangular inferior e superior
    """
def decomposicao_lu(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float)
    
    for k in range(n-1):
        for i in range(k+1, n):
            if U[k, k] == 0:
                raise ValueError("Divisão por zero - matriz singular")
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return L, U

""" Descrição da Função:

Realiza a decomposição de Cholesky de uma matriz simétrica positiva definida.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz simétrica positiva definida (n x n)
        
    Retorna:
    --------
    array:
        Matriz triangular inferior L tal que A = LLᵀ
    """
def decomposicao_cholesky(A):
    n = A.shape[0]
    
    # Verifica se a matriz é quadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz deve ser quadrada")
    
    # Verifica se a matriz é simétrica
    if not np.allclose(A, A.T):
        raise ValueError("A matriz deve ser simétrica")
    
    L = np.zeros_like(A, dtype=float)
    
    for j in range(n):
        # Termo diagonal
        soma = np.sum(L[j, :j]**2)
        termo = A[j, j] - soma
        
        if termo <= 0:
            raise ValueError("A matriz não é positiva definida")
        
        L[j, j] = np.sqrt(termo)
        
        # Termos não diagonais
        for i in range(j+1, n):
            soma = np.sum(L[i, :j] * L[j, :j])
            L[i, j] = (A[i, j] - soma) / L[j, j]
    
    return L

""" Descrição da Função:

Resolve um sistema linear Ax = b usando decomposição LU.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz de coeficientes (n x n)
    b : array_like
        Vetor de termos independentes (n)
        
    Retorna:
    --------
    array:
        Vetor solução x
    """
def resolver_lu(A, b):
    
    L, U = decomposicao_lu(A)
    y = substituicao_sucessiva(L, b)
    x = substituicao_retroativa(U, y)
    return x

""" Descrição da Função:

Resolve um sistema linear Ax = b usando decomposição de Cholesky.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz simétrica positiva definida (n x n)
    b : array_like
        Vetor de termos independentes (n)
        
    Retorna:
    --------
    array:
        Vetor solução x
    """
def resolver_cholesky(A, b): 
    L = decomposicao_cholesky(A)
    y = substituicao_sucessiva(L, b)
    x = substituicao_retroativa(L.T, y)
    return x

""" Descrição da Função:
    
Aplica transformação elementar em uma matriz aumentada.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz de coeficientes
    b : array_like
        Vetor de termos independentes
    linha_mod : int
        Índice da linha a ser modificada
    linha_base : int
        Índice da linha base para a transformação
    fator : float
        Fator multiplicativo
        
    Retorna:
    --------
    tuple:
        (A_modificado, b_modificado, T_elementar)
    """
def matriz_elementar(A, b, linha_mod, linha_base, fator):
    
    n = A.shape[0]
    A = A.astype(float)
    b = b.astype(float)
    
    T = np.eye(n)
    T[linha_mod, linha_base] = fator
    
    A_mod = T @ A
    b_mod = T @ b
    
    return A_mod, b_mod, T

""" Descrição da Função:
    
Eliminação de Gauss com pivotamento parcial.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz de coeficientes
    b : array_like
        Vetor de termos independentes
        
    Retorna:
    --------
    array:
        Vetor solução x
    """
def gauss_pivotamento_parcial(A, b):
    
    n = A.shape[0]
    A = A.astype(float)
    b = b.astype(float)
    
    for k in range(n-1):
        # Pivotamento parcial
        pivo = np.argmax(np.abs(A[k:, k])) + k
        if pivo != k:
            A[[k, pivo]] = A[[pivo, k]]
            b[[k, pivo]] = b[[pivo, k]]
        
        # Eliminação
        for i in range(k+1, n):
            fator = A[i, k] / A[k, k]
            A[i, k:] -= fator * A[k, k:]
            b[i] -= fator * b[k]
    
    return substituicao_retroativa(A, b)

# ==============================================
# MÉTODOS PARA REGRESSÃO LINEAR MÚLTIPLA
# ==============================================

""" Descrição da Função:
    
Realiza regressão linear múltipla usando mínimos quadrados.
    
    Parâmetros:
    -----------
    X : array_like
        Matriz de variáveis independentes (m x n)
    y : array_like
        Vetor de variável dependente (m)
    retornar_modelo : bool, opcional
        Se True, retorna o modelo sklearn (padrão=False)
        
    Retorna:
    --------
    tuple or None:
        - Se retornar_modelo=True: (modelo, coef, intercepto)
        - Caso contrário: imprime os resultados
    """
def regressao_linear_multipla(X, y, retornar_modelo=False):
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    coef = modelo.coef_
    intercepto = modelo.intercept_
    
    # Formata a equação
    termos = [f"{intercepto:.4f}"]
    for i, c in enumerate(coef):
        termos.append(f"{c:.4f}*x{i+1}")
    equacao = "y = " + " + ".join(termos)
    
    print("Modelo de Regressão Linear Múltipla:")
    print(equacao)
    print(f"Coeficiente de determinação (R²): {modelo.score(X, y):.4f}")
    
    if retornar_modelo:
        return modelo, coef, intercepto

""" Descrição da Função:

Plota uma regressão linear múltipla com 2 variáveis independentes em 3D.
    
    Parâmetros:
    -----------
    X : array_like de forma (n, 2)
        Matriz com as duas variáveis independentes (colunas x1 e x2)
    y : array_like de forma (n,)
        Vetor da variável dependente
        
    Retorno:
    --------
    None (exibe o gráfico 3D interativo)
    
    Exemplo:
    --------
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([3, 5, 7, 9])
    >>> plot_regressao_multipla_3d(X, y)
    """
def plot_regressao_multipla_3d(X, y):
    # Verificação de inputs
    if X.shape[1] != 2:
        raise ValueError("X deve ter exatamente 2 colunas para plotagem 3D")
    
    # Criação do modelo
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Preparação da malha para a superfície
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    x1_range = np.linspace(min(x1), max(x1), 20)
    x2_range = np.linspace(min(x2), max(x2), 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Calcula os valores preditos para a superfície
    Y_pred = modelo.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
    
    # Criação da figura 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot dos pontos originais
    ax.scatter(x1, x2, y, c='r', marker='o', depthshade=True, s=50, label='Dados observados')
    
    # Plot da superfície de regressão
    ax.plot_surface(X1, X2, Y_pred, alpha=0.5, cmap='viridis', 
                   edgecolor='none', label='Plano de regressão')
    
    # Configurações do gráfico
    ax.set_xlabel('Variável X1', fontsize=12)
    ax.set_ylabel('Variável X2', fontsize=12)
    ax.set_zlabel('Variável Y', fontsize=12)
    ax.set_title('Regressão Linear Múltipla (3D)\n' +
                f'Equação: y = {modelo.intercept_:.2f} + {modelo.coef_[0]:.2f}x1 + {modelo.coef_[1]:.2f}x2\n' +
                f'R² = {modelo.score(X, y):.4f}', fontsize=14)
    
    # Ângulo de visualização
    ax.view_init(elev=25, azim=45)
    
    # Legenda e grid
    ax.legend()
    ax.grid(True)
    
    # Ajustes finais
    plt.tight_layout()
    plt.show()

# ==============================================
# MÉTODOS PARA INTERPOLAÇÃO POLINOMIAL
# ==============================================

""" Descrição da Função:

Realiza interpolação polinomial para um conjunto de pontos.
    
    Parâmetros:
    -----------
    pontos : array_like
        Matriz 2D onde a primeira linha são os x's e a segunda os y's
        
    Retorna:
    --------
    array:
        Coeficientes do polinômio interpolador [a0, a1, ..., an]
    """
def interpolacao_polinomial(pontos):
    grau = pontos.shape[1]
    if grau == 1:
        a1 = (pontos[1,1] - pontos[1,0]) / (pontos[0,1] - pontos[0,0])
        a0 = pontos[1,0] - a1 * pontos[0,0]
        return np.array([a0, a1])
    else:
        n = grau
        # Cria a matriz de Vandermonde
        V = np.vander(pontos[0,:], increasing=True)
        # Resolve o sistema linear
        coeficientes = np.linalg.solve(V, pontos[1,:])
        return coeficientes


# ==============================================
# MÉTODOS ADICIONAIS
# ==============================================

""" Descrição da Função:

Comparação gráfica dos métodos de integração.
    
    Parâmetros:
    -----------
    f : function
        Função a ser integrada
    a : float
        Limite inferior
    b : float
        Limite superior
    n : int
        Número de subintervalos para visualização
    """
def plot_comparativo_integracao(f, a, b, n=10): 
    x = np.linspace(a, b, 1000)
    y = f(x)
    x_trap = np.linspace(a, b, n+1)
    y_trap = f(x_trap)
    
    plt.figure(figsize=(12, 6))
    
    # Plot da função
    plt.plot(x, y, 'b-', label='Função')
    
    # Área exata
    plt.fill_between(x, y, color='lightgreen', alpha=0.3, label='Área exata')
    
    # Trapézios
    for i in range(n):
        xs = [x_trap[i], x_trap[i+1], x_trap[i+1], x_trap[i]]
        ys = [0, 0, y_trap[i+1], y_trap[i]]
        plt.fill(xs, ys, color='lightblue', alpha=0.5, edgecolor='red')
    
    plt.title('Comparação de Métodos de Integração')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ==============================================
# FUNÇÕES DE REGRESSÃO AVANÇADAS
# ==============================================

""" Descrição da Função:

Regressão polinomial com validações.
    
    Parâmetros:
    -----------
    x : array_like
        Variável independente
    y : array_like
        Variável dependente
    grau : int
        Grau do polinômio
    plot : bool
        Se True, mostra o gráfico
        
    Retorna:
    --------
    array:
        Coeficientes do polinômio
    """
def regressao_polinomial_segura(x, y, grau, plot=False):
    if len(x) != len(y):
        raise ValueError("x e y devem ter o mesmo tamanho")
    if grau >= len(x):
        raise ValueError("Grau deve ser menor que o número de pontos")
    
    X = np.vander(x, grau + 1, increasing=True)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    
    if plot:
        plt.scatter(x, y, label='Dados')
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = sum(c * x_vals**i for i, c in enumerate(coef))
        plt.plot(x_vals, y_vals, 'r', label=f'Polinômio grau {grau}')
        plt.legend()
        plt.show()
    
    return coef

""" Descrição da Função:
    
Plota regressão linear múltipla com 2 variáveis em 3D.
    
    Parâmetros:
    -----------
    X : array_like
        Matriz com 2 colunas de variáveis independentes
    y : array_like
        Vetor da variável dependente
    """
def grafico_regressao_multipla_3d(X, y):
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    # Criar malha para a superfície
    x1_range = np.linspace(min(x1), max(x1), 20)
    x2_range = np.linspace(min(x2), max(x2), 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Y = modelo.intercept_ + modelo.coef_[0]*X1 + modelo.coef_[1]*X2
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pontos originais
    ax.scatter(x1, x2, y, c='r', marker='o', label='Dados')
    
    # Superfície de regressão
    ax.plot_surface(X1, X2, Y, alpha=0.5, color='b', label='Plano de regressão')
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.title('Regressão Linear Múltipla (3D)')
    plt.legend()
    plt.show()

# ==============================================
# MÉTODOS DE ÁLGEBRA LINEAR AVANÇADOS
# ==============================================

""" Descrição da Função: 

Resolve sistema linear usando decomposição LU.
    
    Parâmetros:
    -----------
    L : array_like
        Matriz triangular inferior
    U : array_like
        Matriz triangular superior
    b : array_like
        Vetor de termos independentes
        
    Retorna:
    --------
    array:
        Vetor solução x
    """
def resolver_sistema_lu(L, U, b):
    y = substituicao_sucessiva(L, b)
    x = substituicao_retroativa(U, y)
    return x

""" Descrição da Função:

Decomposição LU com pivotamento parcial.
    
    Parâmetros:
    -----------
    A : array_like
        Matriz a ser decomposta
        
    Retorna:
    --------
    tuple:
        (L, U, P) onde:
        - L: Matriz triangular inferior
        - U: Matriz triangular superior
        - P: Matriz de permutação
    """
def decomposicao_lu_pivotamento(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    
    for k in range(n-1):
        # Pivotamento
        pivo = k + np.argmax(np.abs(U[k:, k]))
        if pivo != k:
            U[[k, pivo]] = U[[pivo, k]]
            P[[k, pivo]] = P[[pivo, k]]
            if k > 0:
                L[[k, pivo], :k] = L[[pivo, k], :k]
        
        # Eliminação
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return L, U, P


# ==============================================
# MÉTODOS DE INTERPOLAÇÃO COMPLETOS
# ==============================================

""" Descrição da Função:

Interpolação pelo método de Lagrange.
    
    Parâmetros:
    -----------
    x : array_like
        Pontos x conhecidos
    y : array_like
        Pontos y conhecidos
    ponto : float, optional
        Ponto para interpolar
    plot : bool, optional
        Se True, plota o gráfico
        
    Retorna:
    --------
    float or function:
        - Se ponto for fornecido: valor interpolado
        - Caso contrário: função interpoladora
    """
def lagrange(x, y, ponto=None, plot=False):
    n = len(x)
    
    def base(i, xi):
        """Polinômio base de Lagrange"""
        prod = 1.0
        for j in range(n):
            if j != i:
                prod *= (xi - x[j]) / (x[i] - x[j])
        return prod
    
    def interpolador(xi):
        """Calcula o valor interpolado"""
        return sum(y[i] * base(i, xi) for i in range(n))
    
    if ponto is not None:
        return interpolador(ponto)
    
    if plot:
        xi = np.linspace(min(x), max(x), 100)
        yi = [interpolador(x) for x in xi]
        
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color='red', label='Pontos conhecidos')
        plt.plot(xi, yi, label='Polinômio de Lagrange')
        plt.legend()
        plt.grid(True)
        plt.title('Interpolação de Lagrange')
        plt.show()
    
    return interpolador


# ==============================================
# MÉTODOS DE INTEGRAÇÃO COMPLETOS
# ==============================================

""" Descrição da Função: 

Integração pelo método de Romberg.
    
    Parâmetros:
    -----------
    f : function
        Função a integrar
    a : float
        Limite inferior
    b : float
        Limite superior
    tol : float, optional
        Tolerância de erro
    max_iter : int, optional
        Máximo de iterações
        
    Retorna:
    --------
    float:
        Valor da integral
    """
def integral_romberg(f, a, b, tol=1e-6, max_iter=10):
    R = np.zeros((max_iter, max_iter))
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))
    
    for i in range(1, max_iter):
        h /= 2
        # Regra do trapézio composta
        soma = sum(f(a + (2*k-1)*h) for k in range(1, 2**(i-1)+1))
        R[i, 0] = 0.5 * R[i-1, 0] + h * soma
        
        # Extrapolação de Richardson
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
        
        # Verifica convergência
        if i > 0 and abs(R[i, i] - R[i-1, i-1]) < tol:
            return R[i, i]
    
    return R[max_iter-1, max_iter-1]


# ==============================================
# FUNÇÕES AUXILIARES COMPLETAS
# ==============================================

""" Descrição da Função:
    Gera tabela de diferenças divididas para Newton.
    
    Parâmetros:
    -----------
    x : array_like
        Pontos x
    y : array_like
        Pontos y
        
    Retorna:
    --------
    array:
        Tabela de diferenças divididas
    """
def tabela_diferencas_divididas(x, y):
    n = len(x)
    tabela = np.zeros((n, n))
    tabela[:,0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            tabela[i,j] = (tabela[i+1,j-1] - tabela[i,j-1]) / (x[i+j] - x[i])
    
    return tabela

""" Descrição da Função:

Imprime tabela formatada de diferenças divididas.
    
    Parâmetros:
    -----------
    x : array_like
        Pontos x
    y : array_like
        Pontos y
    """
def print_tabela_diferencas(x, y):
    
    tabela = tabela_diferencas_divididas(x, y)
    n = len(x)
    
    print("\nTabela de Diferenças Divididas:")
    print("-" * 60)
    print(f"{'x':>10}{'f(x)':>12}", end="")
    for i in range(1, n):
        print(f"{f'D^{i}y':>12}", end="")
    print("\n" + "-" * 60)
    
    for i in range(n):
        print(f"{x[i]:>10.4f}", end="")
        for j in range(i+1):
            print(f"{tabela[i-j,j]:>12.6f}", end="")
        print()