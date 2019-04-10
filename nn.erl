-module(nn).
-compile(export_all).

setup(Layers, NeuronCount, N) ->
	%% Construct training data
	Ins = {randNos(N), randNos(N)},
	Expected = calculateExpected(),
	%% Spawn neurons
	layerSetup(first, Ins, Layers, NeuronCount, Expected),
	ok.

randNos(N) -> [rand:uniform(100) || _ <- lists:seq(1, N)].
randDub(N) -> [rand:uniform() || _ <- lists:seq(1, N)].

calculateExpected({[], []}) -> [];
calculateExpected({[X|Xs], [Y|Ys]}) when 100*math:sin(X)/X > Y -> 
	[1|calculateExpected(Xs, Ys)];
calculateExpected({[X|Xs], [Y|Ys]}) ->
	[0|calculateExpected(Xs, Ys)].

sendLayer([], _Layer) -> ok;
sendLayer([N|Ns], Layer) ->
	N!{nextLayer, Layer},
	sendLayer(Ns, Layer).

firstNeuronSetup(Ins) -> 
	receive {nextLayer, Neurons} -> 
		firstNeuron(Ins, Neurons)
	end.
firstNeuron(Ins, Neurons) ->
	receive {sendS, _S} ->
		sendNextIn(In, Neurons)
	end,
	firstNeuron(Ins, Neurons).

sendNextIn(In, []) ->
	ok;
sendNextIn(In, [N|Ns]) ->
	N!{input, In}
	sendNextIns(In, Ns).

layerSetup(first, {Xs, Ys}, Layers, NeuronCount, Expected) ->
	State1 = spawn(?MODULE, firstNeuronSetup, [X]),
	State2 = spawn(?MODULE, firstNeuronSetup, [Y]),
	Pids = [State1] ++ [State2],
	layerSetup(Layers, NeuronCount, Pids, Expected).
layerSetup(0, _NeuronCount, PrevNeurons) ->
	State1 = spawn(?MODULE, finalNeuron, [PrevNeurons, Expected]),
	State2 = spawn(?MODULE, finalNeuron, [PrevNeurons, Expected]),
	Pids = [State1] ++ [State2],
	sendLayer(PrevNeurons, Pids);
layerSetup(Layers, NeuronCount, PrevNeurons, Expected) ->
	Pids = spawnNeurons(NeuronCount, PrevNeurons),
	sendLayer(PrevNeurons, Pids),
	layerSetup(Layers - 1, NeuronCount, Pids, Expected).

spawnNeurons(0, _PrevNeurons) -> [];
spawnNeurons(NeuronCount, PrevNeurons) -> 
	Pid = spawn(?MODULE, neuronSetup, [PrevNeurons]),
	[Pid|spawnNeurons(NeuronCount-1, PrevNeurons)].

neuronSetup(PrevNeurons) ->
	receive 
		{nextLayer, Neurons} -> neuronMain([], [randDub(length(Neurons))], [randDub(length(Neurons))], PrevNeurons, Neurons)
	end.

neuronMain(Ins, Weights, Bias, PrevNeurons, Neurons) ->
	receive 
		{input, X} -> 
			case length(Ins) =:= length(PrevNeurons) - 1 of 
				true -> 
					Value = calculateOut([X|Ins], Weights, Bias),
					sendOut(Value, Neurons);
					neuronMain([X|Ins], Weights, Bias, PrevNeurons, Neurons);
				_ ->
					neuronMain([X|Ins], Weights, Bias, PrevNeurons, Neurons)
			end;
		{sendS, S} -> calculateChanges(S, Ins, Weights, Bias, PrevNeurons, Neurons)
	end.

calculateOut([], [], _Bias) ->
	0;
calculateOut([A|As], [W|Ws], Bias) ->
	1/(1+math:exp((-calculateOut(As, Ws, Bias, Neurons) - A*W - Bias)).

sendOut(_Val, []) ->
	ok;
sendOut(Val, [N, Ns]) ->
	N!{input, N}.

calculateChanges(S, Ins, Weights, Bias, PrevNeurons, Neurons) ->
	Alpha = 0.1,
	NewW = newW(S, Alpha, Weights),
	NewB = Bias - Alpha*S,
	calculateS(S, Ins, Weights, PrevNeurons),
	neuronMain([], NewW, NewB, PrevNeuron, Neurons).
	
newW(S, Alpha, [], []) -> [];
newW(S, Alpha, [A|As], [W|Ws]) ->
	[(W-Alpha*S*A)|newW(S, Alpha, As, Ws)].

calculateS(_S, [], [], []) -> ok;
calculateS(S, [A|As], [W|Ws], [N|Ns]) ->
	N!{sendS, S*W(1-A)*A},
	calculateS(S, As, Ws, Ns).

finalNeuron(Ins, Weights, Bias, PrevNeurons, [Expected|Others]) ->
	receive
		{input, X} ->
			case length(Ins) =:= length(PrevNeurons) - 1 of
				true -> 
					Value = calculateOut([X|Ins], Weights, Bias),
					% Calculate error
					Error = math:pow((Expected - Value), 2),
					% Calculate S
					S =  % Derivative of LSM etc.
					% Send value of S to previous neurons
					% Calculate new weights + bias
					NewW = newW(S, 0.1, Weights),
					NewB = Bias - 0.1 * S,
					finalNeuron([], NewW, NewB, PrevNeurons, Others);
				_ ->
					finalNeuron([X|Ins], Weights, Bias, PrevNeurons, [Expected|Others])
			end;
		}
	end.
