/*
SQLyog Ultimate v11.11 (64 bit)
MySQL - 5.7.9 : Database - staffattandaceonface
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`staffattandaceonface` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `staffattandaceonface`;

/*Table structure for table `attandance` */

DROP TABLE IF EXISTS `attandance`;

CREATE TABLE `attandance` (
  `attandance_id` int(11) NOT NULL AUTO_INCREMENT,
  `staff_id` int(11) DEFAULT NULL,
  `date` varchar(100) DEFAULT NULL,
  `time` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`attandance_id`)
) ENGINE=MyISAM AUTO_INCREMENT=16 DEFAULT CHARSET=latin1;

/*Data for the table `attandance` */

insert  into `attandance`(`attandance_id`,`staff_id`,`date`,`time`) values (1,2,'2020-10-23','07:36:24'),(2,2,'2020-10-23','07:36:24'),(3,2,'2020-10-23','07:36:25'),(4,2,'2020-10-23','07:36:25'),(5,2,'2020-10-23','07:36:26'),(6,2,'2020-10-23','07:36:26'),(7,2,'2020-10-23','07:36:27'),(8,2,'2020-10-23','07:36:27'),(9,2,'2020-10-23','07:36:28'),(10,2,'2020-10-23','15:31:58'),(11,2,'2020-10-23','15:33:27'),(12,2,'2020-10-23','15:36:32'),(13,2,'2020-10-23','15:37:28'),(14,2,'2020-10-23','15:37:57'),(15,2,'2020-11-01','20:32:54');

/*Table structure for table `login` */

DROP TABLE IF EXISTS `login`;

CREATE TABLE `login` (
  `login_id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(100) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  `usertype` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`login_id`)
) ENGINE=MyISAM AUTO_INCREMENT=4 DEFAULT CHARSET=latin1;

/*Data for the table `login` */

insert  into `login`(`login_id`,`username`,`password`,`usertype`) values (1,'admin','admin','admin');

/*Table structure for table `staff` */

DROP TABLE IF EXISTS `staff`;

CREATE TABLE `staff` (
  `staff_id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  `aadhar` varchar(100) DEFAULT NULL,
  `doj` varchar(100) DEFAULT NULL,
  `image` varchar(1000) DEFAULT NULL,
  `noofinput` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`staff_id`)
) ENGINE=MyISAM AUTO_INCREMENT=4 DEFAULT CHARSET=latin1;

/*Data for the table `staff` */

insert  into `staff`(`staff_id`,`name`,`aadhar`,`doj`,`image`,`noofinput`) values (1,'jk','kh','hkh','static/uploads/4fc9d7e8-1806-41ab-b547-6b98f8cf280dmammootty.jpg',NULL),(2,'jkh','kjh','kjh','static/uploads/81c1edc3-b712-4c0d-9c29-11cce6730a96test.jpg',NULL),(3,'lkh','8768','2020-11-04','static/uploads/9bea8ef6-ba09-4e8c-9191-8855cbb6c797test.jpg','1');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
