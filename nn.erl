-module(nn).
-compile(export_all).

run(TrainingData) ->
	oof.

neuronMain(Ins, Weights, Bias, PrevNeurons, Neuron) ->
	receive 
		{input, X} -> 
			case length(Ins) =:= 1 of 
				true -> 
					Neuron!calculateOut([X|Ins], Weights, Bias),
					neuronMain([X|Ins], Weights, Bias, PrevNeurons, Neuron);
				_ =>
					neuronMain([X|Ins], Weights, Bias, PrevNeurons, Neuron)
			end;
		{sendS, S} -> calculateChanges(S, Ins, Weights, Bias, PrevNeurons, Neuron)
	end.

calculateChanges(S, Ins, Weights, Bias, PrevNeurons, Neuron) ->
	Alpha = 0.1,
	NewW = newW(S, Alpha, Weights),
	NewB = Bias - Alpha*S,
	calculateS(S, Ins, Weights, PrevNeurons),
	neuronMain([], NewW, NewB, PrevNeuron, Neuron).
	
newW(S, Alpha, [], []) -> [];
newW(S, Alpha, [A|As], [W|Ws]) ->
	[(W-Alpha*S*A)|newW(S, Alpha, As, Ws)].

calculateS(_S, [], [], []) -> ok;
calculateS(S, [A, As], [W|Ws], [N|Ns]) ->
	N!{sendS, S*W(1-A)*A},
	calculateS(S, As, Ws, Ns).

neuronServer() ->
	ok.

finalNeuron() ->
	receive
		Out ->
			% Calculate error
			% Send S to previous neuron(s)
			% Re-call server
	end.