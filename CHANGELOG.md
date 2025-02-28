# Change Log #

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2025-02-27

## Added

- TiredBuffalo -- FatigueBandit

### Changed

- Minor fixes for warnings in pytest from gym checker

## [0.2.0] - 2024-12-26

### Added

- BounlessBuffalo -- InfiniteArmedBandit
- DuelingBuffalo -- DuelingBandit
- SymbolicState

### Changed

- BuffaloTrail more closely matches the notion of a bandit.  It was previously unconstrained on two axes 
compared to a traditional bandit (control over future states and dependence on history).  Control over future 
states was removed as dependence on history is interesting in its own right.

## [0.1.0] - 2024-11-29

### Added

- Utility which might help in verifying what an agent is learning
- Citation information

### Changed

- Added repeatability to environments using seeds and enforcing the difference between optimal and 
suboptimal arms
- Added "standard" names for environments.  Not everyone will get my naming scheme.
- Environments use info to return necessary information to derive the solution to the environment.