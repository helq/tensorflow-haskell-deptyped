-- Copyright 2017-2018 Elkin Cruz.
-- Copyright 2017 James Bowen.
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

{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE TypeInType            #-}

module TensorFlow.DepTyped.Tensor (
  Tensor(Tensor),
  Placeholder,
  Renderable,
  Feed(Feed),
  FeedList(NilFeedList,(:~~)),
  render,
  feed
) where

import           GHC.TypeLits (Nat, Symbol)
import           Data.Kind (Type)

import           TensorFlow.Build (Build, MonadBuild)
import           TensorFlow.Tensor (Value)
import qualified TensorFlow.Tensor as TF (Feed, feed, Tensor, render)
import qualified TensorFlow.Types as TF ()

import           TensorFlow.DepTyped.Base (PlaceholderNotInList)
import           TensorFlow.DepTyped.Types (TensorData(TensorData))

--TODO(helq): replace current Tensor definition for following
--data ConstraintDim = DimEq Symbol Symbol
--                   | DimAssign Symbol Nat
--data TensorEnvironment =
--  TensorEnv {
--              placeholders :: [(Symbol, [Dim], Type)],
--              variables    :: [(Symbol, [Dim], Type)],
--              dimRestrictions :: [ConstraintDim]
--            }
--
--data Tensor (s :: [Dim]) (te :: TensorEnvironment) v a where
--  Tensor :: (TF.TensorType a) => TF.Tensor v a -> Tensor s te v a

newtype Tensor (shape :: [Nat]) (placeholders :: [(Symbol, [Nat], Type)]) v a = Tensor (TF.Tensor v a)

type Placeholder name shape t = Tensor shape '[ '(name, shape, t) ] Value t

type Renderable shape t = Tensor shape '[] Build t

newtype Feed (name :: Symbol) (shape :: [Nat]) (a :: Type) = Feed TF.Feed

data FeedList (placeholders :: [(Symbol, [Nat], Type)]) where
  NilFeedList :: FeedList '[]
  (:~~) :: PlaceholderNotInList name phs
        => Feed name shape a
        -> FeedList phs
        -> FeedList ('(name, shape, a) ': phs)

infixr 5 :~~

-- TODO(helq): change [Nat] for [Dim]
render :: forall (shape::[Nat]) (plholders::[(Symbol,[Nat],Type)]) a m.
          MonadBuild m
       => Tensor shape plholders Build a -> m (Tensor shape plholders Value a)
render (Tensor t) = Tensor <$> TF.render t

-- TODO(helq): replace Placeholder for something more general as it used in the non-deptyped `feed`
--             the shape from placeholder could be [Dim] while TensorData must be [Nat], the result
--             is a shape of [Nat] for Feed
feed :: Placeholder name shape a -> TensorData name shape a -> Feed name shape a
feed (Tensor t) (TensorData td) = Feed $ TF.feed t td
