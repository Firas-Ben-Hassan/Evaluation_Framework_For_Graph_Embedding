print (" Select the task ")
print("1) Nodde Classification ")
print("2) Link Prediction") 
select = input("Enter option: ")

# Node Classification 
if select == "1" :
	import evaluate_node_classification
	emb = evaluate_node_classification.run()

# Link Prediction  
if select == "2" :
	import evaluate_link_prediction
	emb = evaluate_link_prediction.evaluateStaticLinkPrediction(Graph Embeddings , 0.8)          # train ratio = 0.8
                
        
            
        
            
       
            
            
            
        
        
