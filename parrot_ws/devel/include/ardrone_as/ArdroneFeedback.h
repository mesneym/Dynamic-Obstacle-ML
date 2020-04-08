// Generated by gencpp from file ardrone_as/ArdroneFeedback.msg
// DO NOT EDIT!


#ifndef ARDRONE_AS_MESSAGE_ARDRONEFEEDBACK_H
#define ARDRONE_AS_MESSAGE_ARDRONEFEEDBACK_H


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
struct ArdroneFeedback_
{
  typedef ArdroneFeedback_<ContainerAllocator> Type;

  ArdroneFeedback_()
    : lastImage()  {
    }
  ArdroneFeedback_(const ContainerAllocator& _alloc)
    : lastImage(_alloc)  {
  (void)_alloc;
    }



   typedef  ::sensor_msgs::CompressedImage_<ContainerAllocator>  _lastImage_type;
  _lastImage_type lastImage;





  typedef boost::shared_ptr< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> const> ConstPtr;

}; // struct ArdroneFeedback_

typedef ::ardrone_as::ArdroneFeedback_<std::allocator<void> > ArdroneFeedback;

typedef boost::shared_ptr< ::ardrone_as::ArdroneFeedback > ArdroneFeedbackPtr;
typedef boost::shared_ptr< ::ardrone_as::ArdroneFeedback const> ArdroneFeedbackConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ardrone_as::ArdroneFeedback_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace ardrone_as

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'ardrone_as': ['/home/ak/Dynamic-Obstacle-ML/Dynamic-Obstacle-PP/parrot_ws/devel/share/ardrone_as/msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'actionlib_msgs': ['/opt/ros/kinetic/share/actionlib_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "5b7da50a022bc1e4f64b32f363fda187";
  }

  static const char* value(const ::ardrone_as::ArdroneFeedback_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x5b7da50a022bc1e4ULL;
  static const uint64_t static_value2 = 0xf64b32f363fda187ULL;
};

template<class ContainerAllocator>
struct DataType< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ardrone_as/ArdroneFeedback";
  }

  static const char* value(const ::ardrone_as::ArdroneFeedback_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
#feedback\n\
sensor_msgs/CompressedImage lastImage  # the last image taken\n\
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

  static const char* value(const ::ardrone_as::ArdroneFeedback_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.lastImage);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ArdroneFeedback_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ardrone_as::ArdroneFeedback_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ardrone_as::ArdroneFeedback_<ContainerAllocator>& v)
  {
    s << indent << "lastImage: ";
    s << std::endl;
    Printer< ::sensor_msgs::CompressedImage_<ContainerAllocator> >::stream(s, indent + "  ", v.lastImage);
  }
};

} // namespace message_operations
} // namespace ros

#endif // ARDRONE_AS_MESSAGE_ARDRONEFEEDBACK_H
