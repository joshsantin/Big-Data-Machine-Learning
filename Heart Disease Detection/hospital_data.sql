-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Mar 15, 2022 at 10:37 PM
-- Server version: 10.4.22-MariaDB
-- PHP Version: 8.1.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `hospital`
--

-- --------------------------------------------------------

--
-- Table structure for table `doctor_data`
--

CREATE TABLE `doctor_data` (
  `Doctor_ID` int(8) NOT NULL,
  `first_name` varchar(20) NOT NULL,
  `last_name` varchar(20) NOT NULL,
  `email_address` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `doctor_data`
--

INSERT INTO `doctor_data` (`Doctor_ID`, `first_name`, `last_name`, `email_address`) VALUES
(1, 'Sam', 'Ross', 'mross@gmai.com'),
(2, 'Aansa', 'Arshad', 'aansaarshad30@gmail.com'),
(3, 'Alan', 'Turing', 'aturing@gmai.com'),
(4, 'Thomas', 'Edison', 'tedison@gmai.com'),
(5, 'Jordan', 'Belfort', 'jbelfort@gmai.com');

-- --------------------------------------------------------

--
-- Table structure for table `notification_api`
--

CREATE TABLE `notification_api` (
  `Doctor_ID` int(8) NOT NULL,
  `api_key` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `notification_api`
--

INSERT INTO `notification_api` (`Doctor_ID`, `api_key`) VALUES
(1, 'o.9yldO7OXXpGlsyDWplPVgMeAmcBqOZka'),
(2, 'o.pLtMHHEZchIPxTOt1Qxx8RfuxWk9sMXF'),
(3, 'o.9yldO7OXXpGlsyDWplPVgMeAmcBqOZkb'),
(4, 'o.9yldO7OXXpGlsyDWplPVgMeAmcBqOZkc'),
(5, 'o.9yldO7OXXpGlsyDWplPVgMeAmcBqOZkd');

-- --------------------------------------------------------

--
-- Table structure for table `patient_data`
--

CREATE TABLE `patient_data` (
  `PatientID` varchar(20) NOT NULL,
  `Age` int(3) NOT NULL,
  `Sex_M` tinyint(1) NOT NULL,
  `RestingBP` float NOT NULL,
  `Cholesterol` float NOT NULL,
  `FastingBS` tinyint(1) NOT NULL,
  `MaxHR` float NOT NULL,
  `Oldpeak` float NOT NULL,
  `ChestPainType_ATA` tinyint(1) NOT NULL,
  `ChestPainType_NAP` tinyint(1) NOT NULL,
  `ChestPainType_TA` tinyint(1) NOT NULL,
  `RestingECG_LVH` tinyint(1) NOT NULL,
  `RestingECG_ST` tinyint(1) NOT NULL,
  `ExerciseAngina_Y` tinyint(1) NOT NULL,
  `ST_Slope_Flat` tinyint(1) NOT NULL,
  `ST_Slope_Up` tinyint(1) NOT NULL,
  `Doctor_ID` int(8) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `patient_data`
--

INSERT INTO `patient_data` (`PatientID`, `Age`, `Sex_M`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`, `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`, `RestingECG_LVH`, `RestingECG_ST`, `ExerciseAngina_Y`, `ST_Slope_Flat`, `ST_Slope_Up`, `Doctor_ID`) VALUES
('a0001', 54, 1, 150, 195, 0, 122, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2),
('a0002', 40, 1, 140, 289, 0, 172, 0, 1, 0, 0, 0, 0, 0, 0, 1, 3),
('a0003', 67, 1, 94, 33, 1, 89, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1),
('a0004', 21, 1, 165, 169, 1, 65, -2, 1, 0, 0, 1, 0, 0, 0, 0, 4),
('a0005', 21, 1, 141, 287, 1, 110, 1, 1, 0, 0, 1, 0, 1, 0, 0, 3),
('a0006', 48, 0, 138, 214, 0, 108, 1.5, 0, 0, 0, 0, 0, 1, 1, 0, 2),
('a0007', 26, 1, 110, 201, 0, 71, 4, 1, 0, 0, 0, 0, 0, 0, 1, 5),
('a0008', 25, 1, 150, 300, 0, 122, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `doctor_data`
--
ALTER TABLE `doctor_data`
  ADD PRIMARY KEY (`Doctor_ID`);

--
-- Indexes for table `notification_api`
--
ALTER TABLE `notification_api`
  ADD PRIMARY KEY (`Doctor_ID`);

--
-- Indexes for table `patient_data`
--
ALTER TABLE `patient_data`
  ADD PRIMARY KEY (`PatientID`),
  ADD KEY `patient_doctor` (`Doctor_ID`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `doctor_data`
--
ALTER TABLE `doctor_data`
  MODIFY `Doctor_ID` int(8) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `notification_api`
--
ALTER TABLE `notification_api`
  MODIFY `Doctor_ID` int(8) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `doctor_data`
--
ALTER TABLE `doctor_data`
  ADD CONSTRAINT `doctor_api` FOREIGN KEY (`Doctor_ID`) REFERENCES `notification_api` (`Doctor_ID`);

--
-- Constraints for table `patient_data`
--
ALTER TABLE `patient_data`
  ADD CONSTRAINT `patient_doctor` FOREIGN KEY (`Doctor_ID`) REFERENCES `doctor_data` (`Doctor_ID`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
