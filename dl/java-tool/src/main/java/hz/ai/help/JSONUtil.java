package hz.ai.help;

import java.io.IOException;
import java.util.List;

import org.codehaus.jackson.annotate.JsonAutoDetect.Visibility;
import org.codehaus.jackson.annotate.JsonMethod;
import org.codehaus.jackson.map.ObjectMapper;
import org.codehaus.jackson.map.SerializationConfig;
import org.codehaus.jackson.map.annotate.JsonSerialize.Inclusion;
import org.codehaus.jackson.type.TypeReference;

 

/**
 * <B>JSON转换工具</B>
 * @author henry
 * @since 2013-11-17
 */
@SuppressWarnings("all")
public class JSONUtil {
	
	private static ObjectMapper mapper = new ObjectMapper();
	
	
	/**
	 * <B>将对象转换成JSON字符串</B>
	 * @param obj 需要转换的对象
	 * @return {@link java.lang.String}
	 * @throws ProcessException
	 */
	public static String uncode(Object obj) throws  Exception{
		try {
			return mapper.writeValueAsString(obj);
		} catch (IOException e) {
			throw new  Exception("Object convert to JSON failed.", e);
		}
	}
	
	/**
	 * <B>将JSON字符串转换成java对象</B>
	 * @param json {@link java.lang.String} }JSON字符串
	 * @param clz 需要转换成的java类型
	 * @return
	 * @throws ProcessException
	 */
	public static <T> T encode(String json,Class<T> clz) throws  Exception {
		try {
			return mapper.readValue(json,clz);
		} catch (IOException e) {
			throw new  Exception("JSON convert to Object failed.", e);
		}
	}

}
