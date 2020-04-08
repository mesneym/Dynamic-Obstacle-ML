// Generated by gencpp from file ardrone_as/ArdroneResult.msg
// DO NOT EDIT!


#ifndef ARDRONE_AS_MESSAGE_ARDRONERESULT_H
#define ARDRONE_AS_MESSAGE_ARDRONERESULT_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <sensor_msgs/CompressedImage.h>

namespace ardrone_as
{
template <class ContainerAllocator>
struct ArdroneResult_
{
  typedef ArdroneResult_<ContainerAllocator> Type;

  ArdroneResult_()
    : allPictures()  {
    }
  ArdroneResult_(const ContainerAllocator& _alloc)
    : allPictures(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::sensor_msgs::CompressedImage_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::sensor_msgs::CompressedImage_<ContainerAllocator> >::other >  _allPictures_type;
  _allPictures_type allPictures;





  typedef boost::shared_ptr< ::ardrone_as::ArdroneResult_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ardrone_as::ArdroneResult_<ContainerAllocator> const> ConstPtr;

}; // struct ArdroneResult_

typedef ::ardrone_as::ArdroneResult_<std::allocator<void> > ArdroneResult;

typedef boost::shared_ptr< ::ardrone_as::ArdroneResult > ArdroneResultPtr;
typedef boost::shared_ptr< ::ardrone_as::ArdroneResult const> ArdroneResultConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ardrone_as::ArdroneResult_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ardrone_as::ArdroneResult_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace ardrone_as

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'ardrone_as': ['/home/ak/Dynamic-Obstacle-ML/parrot_ws/devel/share/ardrone_as/msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'actionlib_msgs': ['/opt/ros/kinetic/share/actionlib_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ardrone_as::ArdroneResult_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_as::ArdroneResult_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_as::ArdroneResult_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
{
  static const char* value()
  {
    return "fd8074274123c31a020701b6010cff19";
  }

  static const char* value(const ::ardrone_as::ArdroneResult_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xfd8074274123c31aULL;
  static const uint64_t static_value2 = 0x020701b6010cff19ULL;
};

template<class ContainerAllocator>
struct DataType< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ardrone_as/ArdroneResult";
  }

  static const char* value(const ::ardrone_as::ArdroneResult_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
#result\n\
sensor_msgs/CompressedImage[] allPictures # an array containing all the pictures taken along the nseconds\n\
\n\
\n\
================================================================================\n\
MSG: sensor_msgs/CompressedImage\n\
# This message contains a compressed image\n\
\n\
Header header        # Header timestamp should be acquisition time of image\n\
                     # Header frame_id should be optical frame of camera\n\
                     # origin of frame should be optical center of camera\n\
                     # +x should point to the right in the image\n\
                     # +y should point down in the image\n\
                     # +z should point into to plane of the image\n\
\n\
string format        # Specifies the format of the data\n\
                     #   Acceptable values:\n\
                     #     jpeg, png\n\
uint8[] data         # Compressed image buffer\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
";
  }

  static const char* value(const ::ardrone_as::ArdroneResult_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.allPictures);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ArdroneResult_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ardrone_as::ArdroneResult_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ardrone_as::ArdroneResult_<ContainerAllocator>& v)
  {
    s << indent << "allPictures[]" << std::endl;
    for (size_t i = 0; i < v.allPictures.size(); ++i)
    {
      s << indent << "  allPictures[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::sensor_msgs::CompressedImage_<ContainerAllocator> >::stream(s, indent + "    ", v.allPictures[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // ARDRONE_AS_MESSAGE_ARDRONERESULT_H
