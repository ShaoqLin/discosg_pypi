from discosg.parser.DualTaskSceneGraphParser import DualTaskSceneGraphParser

def main():
    # Initialize model
    model = DualTaskSceneGraphParser(
        model_path="sqlinn/DiscoSG-Refiner-Large-t5-only", 
        device="cuda", 
        lemmatize=False, 
        lowercase=True
    )
    
    # Prepare image descriptions
    descriptions = [
        "The image captures a bustling urban scene, likely in a European city. The setting appears to be a pedestrian-friendly square or plaza. There are numerous people of various ages and attire walking around, some carrying bags, suggesting shopping or a day out. A few individuals are seated, possibly enjoying a meal or resting. The square is adorned with a decorative fountain in the center, surrounded by potted plants. Overhead, there are power lines and cables, hinting at an urban environment. The architecture of the surrounding buildings suggests a historic or older part of the city.",
        "In the image, a man is seated at a desk, engrossed in his work on a computer. He's wearing a blue shirt and glasses, and his hand is raised to his forehead in a gesture that suggests deep thought or concentration. The desk, cluttered with various items, houses a computer monitor, keyboard, and mouse. The room around him is dimly lit, creating an atmosphere of focus and seriousness. In the background, a window can be seen, adding depth to the scene. The image captures a moment of intense concentration and productivity."
    ]
    
    # Prepare scene graphs to fix
    graph_to_fix = {
        descriptions[0]: "( city , is , bustling ) , ( city , is , european ) , ( setting , is , pedestrian-friendly ) , ( setting , is , square ) , ( people , carry , bags ) , ( people , is , walking ) , ( individuals , is , seated ) , ( fountain , in center of , square ) , ( fountain , is , decorative ) , ( plants , is , potted ) , ( plants , surround , fountain ) , ( cables , is , overhead ) , ( power lines , is , overhead ) , ( buildings , surround , city ) , ( city , is , historic ) , ( city , is , older )",
        descriptions[1]: "( man , sit at , desk ) , ( man , work on , computer ) , ( hand , lift to , forehead ) , ( man , have , hand ) , ( man , wear , glasses ) , ( shirt , is , blue ) , ( desk , house , monitor ) , ( desk , house , mouse ) , ( desk , is , cluttered ) , ( monitor , is , computer ) , ( man , in , room ) , ( room , is , dimly lit ) , ( window , in , background ) , ( image , capture , concentration ) , ( image , capture , productivity ) , ( productivity , is , intense )"
    }
    
    # Execute parsing
    outputs = model.parse(
        descriptions=descriptions,
        graph_to_fix=graph_to_fix,
        batch_size=2,
        task="delete_before_insert"
    )
    
    # View results
    print("Parsing results:")
    print(outputs)
    print("\nOutput keys:")
    print(outputs.keys())

if __name__ == "__main__":
    main()