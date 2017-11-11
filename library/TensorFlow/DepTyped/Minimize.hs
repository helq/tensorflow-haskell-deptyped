-- Copyright 2017 Elkin Cruz.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE DataKinds      #-}
{-# LANGUAGE RankNTypes     #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeInType     #-}

module TensorFlow.DepTyped.Minimize (
  minimizeWith
) where

import            GHC.TypeLits (Nat, Symbol)
import            Data.Kind (Type)

import qualified TensorFlow.Minimize as TF (minimizeWith, Minimizer)
import qualified TensorFlow.Gradient as TF (GradientCompatible)
import qualified TensorFlow.Variable as TF (Variable)
import           TensorFlow.Build (MonadBuild)

import           TensorFlow.DepTyped.Tensor (Tensor(Tensor))
import           TensorFlow.DepTyped.Output (ControlNode(ControlNode))

minimizeWith :: forall (m :: Type -> Type) a (v :: Type -> Type) (phs::[(Symbol,[Nat],Type)]) (shape::[Nat]).
                (MonadBuild m, TF.GradientCompatible a)
             => TF.Minimizer a
             -> Tensor shape phs v a
             -> [TF.Variable a]
             -> m (ControlNode phs)
minimizeWith minimizer (Tensor t) vars = ControlNode <$> TF.minimizeWith minimizer t vars
