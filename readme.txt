Market1501

	1. data prepare
		python3 prepare --Market
	2. train
		python3 train_market.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir "/home/huangpg/st-reid/dataset/market_rename/"
	3. test
		python3 test_st_market.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_market_e --test_dir "/home/huangpg/st-reid/dataset/market_rename/" 
	4. st model
		python3 gen_st_model.py --name ft_ResNet50_pcb_market_e  --data_dir "/home/huangpg/st-reid/dataset/market_rename/"
	5. evaluate
		python3 evaluate_st.py --name ft_ResNet50_pcb_market_e 
	6. re-rank
		python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_market_e
		python3 evaluate_rerank_market.py --name ft_ResNet50_pcb_market_e

Duke:

	1. data prepare 
		python3 prepare --Duke
	2. train
		python3 train_duke.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_duke_e --erasing_p 0.5 --train_all --data_dir "/home/huangpg/st-reid/dataset/DukeMTMC_prepare/"
	3. test
		python3 test_st_duke.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_duke_e --test_dir "/home/huangpg/st-reid/dataset/DukeMTMC_prepare/" 
	4. st model
		python3 gen_st_model_duke.py --name ft_ResNet50_pcb_duke_e  --data_dir "/home/huangpg/st-reid/dataset/DukeMTMC_prepare/"
	5. evaluate
		python3 evaluate_st.py --name ft_ResNet50_pcb_duke_e 
	6. re-rank
		python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_duke_e
		python3 evaluate_rerank_duke.py --name ft_ResNet50_pcb_duke_e




