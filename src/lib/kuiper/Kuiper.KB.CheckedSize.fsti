module Kuiper.KB.CheckedSize

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT

(* Checked SizeT arithmetic for ABI-facing entrypoints.  Each operation emits
   a runtime guard before the machine operation and returns its mathematical
   result, allowing callers to establish their ordinary Kuiper size
   predicates inside the verified surface. *)

inline_for_extraction noextract
fn mul
  (x : szp)
  (y : sz)
  norewrite
  requires emp
  returns z : sz
  ensures pure (SZ.v z == SZ.v x * SZ.v y /\
                SZ.fits (SZ.v x * SZ.v y))

inline_for_extraction noextract
fn mulp
  (x y : szp)
  norewrite
  requires emp
  returns z : szp
  ensures pure (SZ.v z == SZ.v x * SZ.v y /\
                SZ.fits (SZ.v x * SZ.v y))

inline_for_extraction noextract
fn mulp3
  (x y z : szp)
  norewrite
  requires emp
  returns r : szp
  ensures pure (SZ.v r == SZ.v x * SZ.v y * SZ.v z /\
                SZ.fits (SZ.v x * SZ.v y * SZ.v z))

inline_for_extraction noextract
fn mulp4
  (w x y z : szp)
  norewrite
  requires emp
  returns r : szp
  ensures pure (SZ.v r == SZ.v w * SZ.v x * SZ.v y * SZ.v z /\
                SZ.fits (SZ.v w * SZ.v x * SZ.v y * SZ.v z))

inline_for_extraction noextract
fn mulp5
  (v w x y z : szp)
  norewrite
  requires emp
  returns r : szp
  ensures pure
    (SZ.v r == SZ.v v * SZ.v w * SZ.v x * SZ.v y * SZ.v z /\
     SZ.fits (SZ.v v * SZ.v w * SZ.v x * SZ.v y * SZ.v z))

inline_for_extraction noextract
fn add
  (x y : sz)
  norewrite
  requires emp
  returns z : sz
  ensures pure (SZ.v z == SZ.v x + SZ.v y /\
                SZ.fits (SZ.v x + SZ.v y))

inline_for_extraction noextract
fn addp
  (x : szp)
  (y : sz)
  norewrite
  requires emp
  returns z : szp
  ensures pure (SZ.v z == SZ.v x + SZ.v y /\
                SZ.fits (SZ.v x + SZ.v y))

inline_for_extraction let () = ()
