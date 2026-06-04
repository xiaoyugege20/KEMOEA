classdef KEMOEA< ALGORITHM
% <multi> <real/binary> <large/none> <constrained/none> <sparse>
% Evolutionary algorithm for sparse multi-objective optimization problems

%------------------------------- Reference --------------------------------
% Y. Tian, X. Zhang, C. Wang, and Y. Jin, An evolutionary algorithm for
% large-scale sparse multi-objective optimization problems, IEEE
% Transactions on Evolutionary Computation, 2020, 24(2): 380-393.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Population initialization
            % Calculate the fitness of each decision variable
           populationmask = UniformPoint(Problem.N,Problem.D,'Latin') > 0.5;
           REAL    = ~strcmp(Problem.encoding,'binary');
           if REAL
              Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
           else
               Dec = ones(Problem.N,Problem.D);
           end
          Dec1=Dec;
          Population = SOLUTION(populationmask.*Dec);          
         s1=0.5;
         s2=0.5;
         index = [];
%          if REAL
%             for i=1:Problem.D
%                 index = [index,i];
%             end
%             descore =laplacian_score(Population.decs, Problem.D*0.1, 0.2, index);
%             [b,idx] = sort(descore);
% %          [b,idx] = sort(descore);         
%             featureselectionMask =zeros(1,Problem.D);
%             featureselectionMask(1,idx(1:Problem.D*0.1)) = ones(1,Problem.D*0.1);
%          else
             for i=1:Problem.D
                index = [index,i];
             end
             descore =laplacian_score(Population.decs, ceil(Problem.D/10), 0.2, index);
             [b,idx] = sort(descore);
             featureselectionMask =zeros(1,Problem.D);
             featureselectionMask(1,idx(1:ceil(Problem.D/10))) = ones(1,ceil(Problem.D/10));
%          end
%           OffspringMask = repmat(OffspringMask,Problem.N,1);
% %    OffDec = sparsesbx(Population.decs,populationmask,Problem.N,Problem.D);
%         OffDec = OperatorGA(Population.decs);
%         Offspring=SOLUTION(OffspringMask.*OffDec);
%         [Population1,~,~,Next] = Realdecenvironmentalselection([Population,Offspring],Problem.N);
%          populationmask1 = [populationmask;OffspringMask];
%          populationmask1 = populationmask1(Next',:);
          populationmask1=populationmask;
          Population1 = Population;         
         A = [];
         
         populationmask2=[];
         for i = 1:Problem.D
                Solution = [];
                Mask1 = zeros(1,Problem.D);
                Mask1(1,i) = 1;
                Mask2 = zeros(1,Problem.D);
                  for j = 1 : 1+4*REAL
                      if REAL
%                          Dec = unifrnd(repmat(Problem.lower,1,1),repmat(Problem.upper,1,1));
                         Dec = repmat(Problem.lower,1,1)+(j-1)/4*(repmat(Problem.upper,1,1)-repmat(Problem.lower,1,1));
                         solution = SOLUTION(Dec.*Mask1);
                      else
                         Dec = ones(1,Problem.D);
                         solution = SOLUTION(Dec.*Mask1);
                      end
                      Solution = [Solution, solution];
                  end
                solution = SOLUTION(Dec.*Mask2);
                populationmask2 = [populationmask2;repmat(Mask1,1+4*REAL,1);Mask2];
                Population = [Solution,solution];
                A = [A,Population];
                Fitness    =  NDSort([Population.objs,Population.cons],inf);
                temp(i,:) = Fitness;
         end
         Population2 =A;
         guidedsolution = zeros(1,Problem.D);  
         for i = 1:Problem.D
                c = unique(temp(i,:));
                if(size(c,2) == 2+4*REAL&&temp(i,2+4*REAL)==min(temp(i,:)))
                    guidedsolution(i) = 0;
                elseif(size(c,2) == 2+4*REAL)
                    guidedsolution(i) = 1;
                else
                    guidedsolution(i) = 1;
                end
         end
         Population = Population1;
         Dec = Dec1;
         populationmask = populationmask1;
         [Population,Dec,populationmask,FrontNo] = environmentalselection1(Population,Dec,populationmask,Problem.N);
         MaxP = false(20,Problem.D);
         MinP = MaxP;
         s1 = 1;
        while Algorithm.NotTerminated(Population)
%             [MaxP,MinP,Nonzero] = posmining(logical(Population(FrontNo==1).decs),MaxP,MinP,20);
             MatingPool = TournamentSelection(2,2*Problem.N,FrontNo);
             
%              [OffDec,OffMask] = pmmoeaoperator(Dec(MatingPool,:),populationmask(MatingPool,:),MaxP(:,Nonzero),MinP(:,Nonzero),Nonzero,Population.decs);
             [OffDec,OffMask] = mixoperator(Dec(MatingPool,:),populationmask(MatingPool,:),guidedsolution,featureselectionMask,Population.decs,s1,REAL);
%              [OffDec,OffMask] = mixoperator(OffDec,OffMask,guidedsolution,featureselectionMask,Population.decs,s1);
             Offspring = SOLUTION(OffDec.*OffMask);
             [Population,Dec,populationmask,FrontNo,Next] = environmentalselection1([Population,Offspring],[Dec;OffDec],[populationmask;OffMask],Problem.N);
             s1 = sum(Next(Problem.N+1:end))/Problem.N;
             if(mod(Problem.FE,floor(Problem.maxFE/3))==0)
             for i=1:Problem.D
                index = [index,i];
             end
             descore =laplacian_score(Population.decs, ceil(Problem.D/10), 0.2, index);
             [~,idx] = sort(descore);
             featureselectionMask =zeros(1,Problem.D);
             featureselectionMask(1,idx(1:ceil(Problem.D/10))) = ones(1,ceil(Problem.D/10));
             index = [];
             end
       end
        end
    end
end