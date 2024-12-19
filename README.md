# Monitoramento com YOLO e DeepFace

Este projeto realiza monitoramento em tempo real utilizando o modelo YOLO para detecção de objetos e o DeepFace para análise de emoções e idades. Adicionalmente, calcula a distância e a velocidade de veículos, gerando estatísticas sobre os eventos capturados.

## Funcionalidades

- **Detecção de objetos**: Identifica veículos em vídeo utilizando o YOLOv8.
- **Cálculo de velocidade**: Mede a velocidade aproximada dos veículos com base na distância e no tempo entre frames.
- **Análise facial**: Detecta rostos e identifica emoções dominantes e idades utilizando o DeepFace.
- **Estatísticas**: Gera dados sobre as emoções mais detectadas e idades encontradas.

## Tecnologias utilizadas

- **YOLOv8**: Para detecção de objetos.
- **DeepFace**: Para análise de emoções e idades.
- **OpenCV**: Para manipulação de vídeo e processamento de imagem.

## Configuração

### Dependências

Certifique-se de que as seguintes bibliotecas estão instaladas:

- `ultralytics`
- `deepface`
- `opencv-python`
- `numpy`

Instale as dependências executando o seguinte comando:

```bash
pip install ultralytics deepface opencv-python numpy
```

### Configuração de hardware

Este projeto requer uma câmera conectada ao computador ou um arquivo de vídeo pré-existente. Certifique-se de que o dispositivo esteja configurado corretamente.

## Uso

1. Clone o repositório:

   ```bash
   [git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio](https://github.com/MSouza27/Monitoramento-com-YOLO-e-DeepFace)
   ```

2. Execute o script principal:

   ```bash
   python __init__.py
   ```

3. Visualize a saída em tempo real na janela de vídeo. Pressione `ESC` para sair.

## Estrutura do projeto

- **`monitoramento.py`**: Script principal que implementa todas as funcionalidades.
- **Requisitos**: Listados no arquivo `requirements.txt` (opcional para uso em ambientes virtuais).

## Resultados esperados

- **Tela de monitoramento**: Exibe os objetos detectados com caixas delimitadoras, IDs únicos, distâncias e velocidades estimadas.
- **Estatísticas de emoções**: Mostra as emoções predominantes e distribuições etárias após a execução do script.

## Contribuição

Sinta-se à vontade para abrir issues e pull requests para melhorias. Feedbacks são bem-vindos!

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

Desenvolvido por Magno Santos.

