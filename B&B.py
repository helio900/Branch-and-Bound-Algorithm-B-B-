from mip import Model, xsum, CBC, MAXIMIZE, CONTINUOUS
import os
import numpy as np

class BB():
    
    # essa é a função principal que coordena a execução do algoritmo. 
    # ela solicita ao usuário que escolha entre inserir os dados via arquivo ou fechar o programa. 
    # em seguida, chama as funções apropriadas para ler os dados do arquivo e executar o algoritmo Branch and Bound.
    def func_chefe(po, nomeArquivo):
        
        FuncaoObjetivaCoeficientes = []
        RestricaoVarCoeficientes = []
        LadoDireitoCoeficientes = []
        VariaveisQuantidade : int = 0
        restricoes : int = 0
        
        try:
            VariaveisQuantidade, restricoes, FuncaoObjetivaCoeficientes = po.lerValoresArquivo(nomeArquivo, RestricaoVarCoeficientes, LadoDireitoCoeficientes)

        except FileNotFoundError as e:
            print(f"Erro ao ler o arquivo: {e}")
            return
        
        po.ramificar(VariaveisQuantidade, FuncaoObjetivaCoeficientes, RestricaoVarCoeficientes, LadoDireitoCoeficientes)
        
        print("")
        print("Melhor solução possível")
        print(f"Valor da Função Objetivo-> {po.otimaSolucao}")
        print("")

    # inicialização da classe BranchAndBound
    # funcao init para criar o objeto da classe
    def __init__(po):
        
        po.valorMaisProximo = 0.5
        po.original = 0  
        po.otimaSolucao = 0

    # lê os valores do arquivo
    # pegamos da internet e adaptamos para o nosso código
    def lerValoresArquivo(po, nomeArquivo, RestricaoVarCoeficientes, LadoDireitoCoeficientes):
        
        if not os.path.exists(nomeArquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {nomeArquivo}")
        
        with open(nomeArquivo, "r") as arquivo:
            VariaveisQuantidade, restricoes = map(int, arquivo.readline().split())
            FuncaoObjetivaCoeficientes = list(map(int, arquivo.readline().split()))
            
            for _ in range(restricoes):
                coeficientes_e_b = list(map(int, arquivo.readline().split()))
                RestricaoVarCoeficientes.append(coeficientes_e_b[:-1])
                LadoDireitoCoeficientes.append(coeficientes_e_b[-1])
        
        return VariaveisQuantidade, restricoes, FuncaoObjetivaCoeficientes

    # cria o modelo de programação linear a ser resolvido
    # tirado dos exemplos práticos do professor Teobaldo
    def criaModelo(po, VariaveisQuantidade: int, FuncaoObjetivaCoeficientes: list[int], RestricaoVarCoeficientes: list[int], LadoDireitoCoeficientes: list[int]):

        modelo = Model(sense=MAXIMIZE, solver_name=CBC)

        x = [modelo.add_var(var_type=CONTINUOUS, lb=0, ub=1, name="x_" + str(i)) for i in range(VariaveisQuantidade)]
        modelo.objective = xsum(FuncaoObjetivaCoeficientes[i] * x[i] for i in range(VariaveisQuantidade))
        
        for j in range(len(LadoDireitoCoeficientes)):
            modelo += xsum(RestricaoVarCoeficientes[j][i] * x[i] for i in range(VariaveisQuantidade)) <= LadoDireitoCoeficientes[j]

        return modelo


    # a função implementa o algoritmo Branch and Bound.
    # ela cria modelos de programação linear com base nas ramificações do problema original, adicionando restrições que dividem o espaço de soluções em subespaços menores
    def ramificar(po, VariaveisQuantidade: int, FuncaoObjetivaCoeficientes: list[int], RestricaoVarCoeficientes: list[int], LadoDireitoCoeficientes: list[int]):        
        
        # modelos que serão resolvidos
        fila = [po.criaModelo(VariaveisQuantidade, FuncaoObjetivaCoeficientes, RestricaoVarCoeficientes, LadoDireitoCoeficientes)]  

        while fila:
            modo, solucaoVars, objetivo = po.fronteira(fila[0])

            if modo in ["inviavel", "limite"]:
                fila.pop(0)
            
            elif modo == "fracao":
                varBranch = solucaoVars[po.valorAproximado([i.x for i in solucaoVars], po.valorMaisProximo)]
                originalModelo = fila.pop(0)
                
                modelo1Novo = originalModelo.copy()
                modelo1Novo += varBranch == 0
                fila.append(modelo1Novo)
                
                modelo2Novo = originalModelo.copy()
                modelo2Novo += varBranch == 1
                fila.append(modelo2Novo)

            elif modo == "integralidade":
                if objetivo is not None and objetivo > po.original:
                    po.original = objetivo
                    po.otimaSolucao = objetivo
                
                fila.pop(0)

    # resolve um modelo de programação linear e determina se a solução é inviável, atingiu o limite superior estabelecido ou contém variáveis fracionárias
    def fronteira(po, modelo): 
        
        modelo.optimize()
        
        variavelFracionaria = False
        solucaoVars = modelo.vars
        objetivo = modelo.objective_value

        if not objetivo: 
            return 'inviavel', [], None

        if objetivo <= po.original: 
            return 'limite', [], None
        
        # verifica se alguma variável é fracionária
        for var in solucaoVars: 
            if var.x != int(var.x):
                variavelFracionaria = True
                break
        
        # retorna integralidade, caso não seja fracionário

        if not variavelFracionaria:
            print("Encontrada integralidade")
            return 'integralidade', solucaoVars, objetivo

        # caso não seja nenhuma opção anterior, retorna fracionário para poder ser ramificado
        return 'fracao', solucaoVars, objetivo
    
    # retorna o valor mais próximo que está na lista, por meio do valor que foi informado
    def valorAproximado(po, array, valor): 
        
        array = np.asarray(array)
        valorEncontrado = np.absolute(array - valor)
        valorEncontrado = valorEncontrado.argmin()
        
        return valorEncontrado 

if __name__ == "__main__":
    
    print("Programa de Branch and Bound - Pesquisa Operacional")
    print("Feito por Hélio e Thomas")
    print("")
    
    nomeArquivo = input("Digite o nome do arquivo para ser feito o branch and bound: ") 
    
    rc = BB()
    rc.func_chefe(nomeArquivo)