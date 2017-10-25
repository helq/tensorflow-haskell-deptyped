-- Copyright 2016 TensorFlow authors.
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

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.Examples.MNISTDeptyped.Parse (
  MNIST,
  drawMNIST,
  readMNISTSamples,
  readMNISTLabels,
  readMessageFromFileOrDie
) where

import Control.Monad (when)
import Data.Binary.Get (Get, runGet, getWord32be, getLazyByteString)
import Data.ByteString.Lazy (toStrict, readFile)
import Data.List.Split (chunksOf)
import Data.Monoid ((<>))
import Data.ProtoLens (Message, decodeMessageOrDie)
import Data.Text (Text)
import Data.Word (Word8, Word32)
import Prelude hiding (readFile)
import qualified Codec.Compression.GZip as GZip
import qualified Data.ByteString.Lazy as L
import qualified Data.Text as Text
import qualified Data.Vector as V

--import Data.Vector.Sized (Vector, toSized)
--import qualified Data.Vector.Sized as VS (splitAt)
--import GHC.TypeLits (type (*), type (-), KnownNat, natVal)
--import Data.Proxy (Proxy(Proxy))

-- | Utilities specific to MNIST.
--type MNISTDep = Vector (28*28) Word8
type MNIST = V.Vector Word8

-- | Produces a unicode rendering of the MNIST digit sample.
drawMNIST :: MNIST -> Text
drawMNIST = chunk . block
  where
    --block_ :: MNISTDep -> Text
    ----block_ :: Vector (28*28) Word8 -> Text
    --block_ = block' (Proxy :: Proxy (28*28))
    --  where block' :: forall proxy n. KnownNat n => proxy n -> Vector n Word8 -> Text
    --        block' p v = let vsize = natVal p in
    --                         case vsize of
    --                           0 -> ""
    --                           n_ -> let (h,t) = (VS.splitAt v :: (Vector 1 Word8, Vector (n-1) Word8)) in
    --                                     "  " <> block' (Proxy :: Proxy (n-1)) t
    block :: V.Vector Word8 -> Text
    block (V.splitAt 1 -> ([0], xs)) = "  " <> block xs
    block (V.splitAt 1 -> ([n], xs)) = c `Text.cons` c `Text.cons` block xs
      where c = "\9617\9618\9619\9608" !! fromIntegral (n `div` 64)
    block (V.splitAt 1 -> _)   = ""
    chunk :: Text -> Text
    chunk "" = "\n"
    chunk xs = Text.take (28*2) xs <> "\n" <> chunk (Text.drop (28*2) xs)

-- | Check's the file's endianess, throwing an error if it's not as expected.
checkEndian :: Get ()
checkEndian = do
    magic <- getWord32be
    when (magic `notElem` ([2049, 2051] :: [Word32])) $
        fail "Expected big endian, but image file is little endian."

-- | Reads an MNIST file and returns a list of samples.
readMNISTSamples :: FilePath -> IO [MNIST]
readMNISTSamples path = do
    raw <- GZip.decompress <$> readFile path
    return $ runGet getMNIST raw
  where
    getMNIST :: Get [MNIST]
    getMNIST = do
        checkEndian
        -- Parse header data.
        cnt  <- fromIntegral <$> getWord32be
        rows <- fromIntegral <$> getWord32be
        cols <- fromIntegral <$> getWord32be
        -- Read all of the data, then split into samples.
        pixels <- getLazyByteString $ fromIntegral $ cnt * rows * cols
        return $ V.fromList <$> chunksOf (rows * cols) (L.unpack pixels)

-- | Reads a list of MNIST labels from a file and returns them.
readMNISTLabels :: FilePath -> IO [Word8]
readMNISTLabels path = do
    raw <- GZip.decompress <$> readFile path
    return $ runGet getLabels raw
  where getLabels :: Get [Word8]
        getLabels = do
            checkEndian
            -- Parse header data.
            cnt <- fromIntegral <$> getWord32be
            -- Read all of the labels.
            L.unpack <$> getLazyByteString cnt

readMessageFromFileOrDie :: Message m => FilePath -> IO m
readMessageFromFileOrDie path = do
    pb <- readFile path
    return $ decodeMessageOrDie $ toStrict pb

-- TODO: Write a writeMessageFromFileOrDie and read/write non-lethal
--             versions.
